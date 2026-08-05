"""Minimal LTX-2.3 video server for Idea Rank (mirrors wan_vast_min_server.py).
Resident DistilledPipeline (fp8-cast) -> generate + multi-frame extend + VideoMAE
embed. HTTP: /health, /generate/enqueue, /jobs/{id}(+/videos/0[/keyframe]),
/extend/enqueue, /embed."""
import argparse, base64, json, os, queue, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import torch

EMBED_FRAMES = 16
STATE = None


def gpu_status():
    try:
        free, total = torch.cuda.mem_get_info()
        return {"vram_used_gb": round((total - free) / 1e9, 1),
                "vram_total_gb": round(total / 1e9, 1)}
    except Exception:
        return {}


class Runtime:
    def __init__(self, distilled, upsampler, gemma, offload, embed_model, out_dir):
        self.distilled, self.upsampler, self.gemma = distilled, upsampler, gemma
        self.offload, self.embed_model = offload, embed_model
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.pipe = None
        self.gen_lock = threading.Lock()
        self._embedder = self._embed_proc = None
        self._embed_lock = threading.Lock()
        self.loaded = False; self.load_error = ""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def preload(self):
        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_pipelines.utils.types import OffloadMode
        from ltx_pipelines.utils.quantization_factory import QuantizationKind
        quant = QuantizationKind("fp8-cast").to_policy(checkpoint_path=self.distilled)
        self.pipe = DistilledPipeline(
            distilled_checkpoint_path=self.distilled,
            spatial_upsampler_path=self.upsampler, gemma_root=self.gemma,
            loras=(), quantization=quant, offload_mode=OffloadMode(self.offload))
        self.loaded = True
        print("LTX pipeline loaded", flush=True)

    def generate(self, prompt, seed, height, width, num_frames, frame_rate, images, out_path):
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.media_io import encode_video
        tiling = TilingConfig.default()
        chunks = get_video_chunks_number(num_frames, tiling)
        # grad mode is THREAD-LOCAL: the worker thread defaults to grad-enabled,
        # which makes the pipeline's internal inference tensors "cannot be saved
        # for backward". Disable grad in this thread for the whole generation.
        with self.gen_lock, torch.no_grad():
            video, audio = self.pipe(
                prompt=prompt, seed=seed, height=height, width=width,
                num_frames=num_frames, frame_rate=frame_rate, images=images,
                tiling_config=tiling, enhance_prompt=False)
            encode_video(video=video, fps=frame_rate, audio=audio,
                         output_path=str(out_path), video_chunks_number=chunks)

    def tail_conditioning(self, src_mp4, n):
        """Extract the last n frames of src_mp4 as pngs, pinned at frames 0..n-1
        of the new clip -> multi-frame (video) extension conditioning."""
        import imageio.v3 as iio
        from ltx_pipelines.utils.args import ImageConditioningInput
        frames = list(iio.imiter(str(src_mp4), plugin="pyav"))
        tail = frames[-n:] if len(frames) >= n else frames
        d = self.out_dir / f"cond_{int(time.time()*1000)}"; d.mkdir(parents=True, exist_ok=True)
        imgs = []
        for i, fr in enumerate(tail):
            p = d / f"{i:03d}.png"
            iio.imwrite(str(p), fr)
            imgs.append(ImageConditioningInput(path=str(p), frame_idx=i, strength=1.0))
        return imgs, len(tail)

    def _load_embedder(self):
        from transformers import AutoImageProcessor, AutoModel
        if self._embedder is None:
            self._embed_proc = AutoImageProcessor.from_pretrained(self.embed_model)
            m = AutoModel.from_pretrained(self.embed_model)
            if self.device == "cuda":
                m = m.to(self.device)
            self._embedder = m.eval()
        return self._embedder, self._embed_proc

    def _sample_frames(self, mp4_path, count):
        import imageio.v3 as iio, numpy as np
        frames = list(iio.imiter(str(mp4_path), plugin="pyav"))
        if not frames:
            return []
        if len(frames) <= count:
            idx = list(range(len(frames)))
        else:
            idx = [int(round(i * (len(frames) - 1) / (count - 1))) for i in range(count)]
        return [np.asarray(frames[i]) for i in idx]

    def embed_path(self, mp4_path):
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
            vec = out.last_hidden_state.float().mean(dim=1).squeeze(0).cpu().tolist()
        return {"embedding": vec, "dims": len(vec), "model": self.embed_model}


class Job:
    def __init__(self, jid, payload):
        self.id = jid; self.payload = payload
        self.status = "queued"; self.error = ""
        self.mp4 = ""; self.keyframe = ""; self.meta = {}


class AppState:
    def __init__(self, runtime):
        self.rt = runtime
        self._jobs = {}; self._counter = 0
        self._lock = threading.Lock()
        self._q = queue.Queue()
        threading.Thread(target=self._loop, name="ltx-worker", daemon=True).start()

    def enqueue(self, payload):
        with self._lock:
            self._counter += 1
            j = Job(self._counter, payload); self._jobs[j.id] = j
        self._q.put(j.id); return j

    def get(self, jid):
        with self._lock:
            return self._jobs.get(int(jid))

    def status(self):
        with self._lock:
            counts = {}
            for j in self._jobs.values():
                counts[j.status] = counts.get(j.status, 0) + 1
            q = sum(1 for j in self._jobs.values() if j.status == "queued")
            r = sum(1 for j in self._jobs.values() if j.status == "running")
        return {"queue_size": q, "running_jobs": r, "status_counts": counts}

    def _loop(self):
        while True:
            jid = self._q.get()
            job = self.get(jid)
            if job is None:
                continue
            job.status = "running"
            t0 = time.time()
            try:
                self._run(job)
                job.status = "done"
            except Exception as exc:
                job.status = "error"; job.error = f"{type(exc).__name__}: {exc}"
                print(f"job {jid} failed: {job.error}", flush=True)
            job.meta["seconds"] = round(time.time() - t0, 1)

    def _run(self, job):
        p = job.payload
        rt = self.rt
        prompt = str(p.get("prompt") or "")
        seed = int(p.get("seed") or 42)
        height = int(p.get("height") or 512); width = int(p.get("width") or 768)
        num_frames = int(p.get("num_frames") or 97)
        frame_rate = float(p.get("frame_rate") or 24.0)
        images = []
        extend_n = 0
        if p.get("extend_from_b64"):
            src = rt.out_dir / f"src_{job.id}.mp4"
            src.write_bytes(base64.b64decode(p["extend_from_b64"]))
            images, extend_n = rt.tail_conditioning(src, int(p.get("extend_frames") or 16))
        out = rt.out_dir / f"{job.id:06d}.mp4"
        rt.generate(prompt, seed, height, width, num_frames, frame_rate, images, out)
        job.mp4 = str(out); job.meta["extend_frames"] = extend_n
        # keyframe png (last frame) for the gallery poster
        try:
            import imageio.v3 as iio
            frames = list(iio.imiter(str(out), plugin="pyav"))
            if frames:
                kf = rt.out_dir / f"{job.id:06d}.png"; iio.imwrite(str(kf), frames[-1])
                job.keyframe = str(kf)
        except Exception as exc:
            print(f"keyframe failed job {job.id}: {exc}", flush=True)


def _json(h, code, obj):
    b = json.dumps(obj).encode()
    h.send_response(code); h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(b))); h.end_headers(); h.wfile.write(b)


def _file(h, path, ctype):
    data = Path(path).read_bytes()
    h.send_response(200); h.send_header("Content-Type", ctype)
    h.send_header("Content-Length", str(len(data))); h.end_headers(); h.wfile.write(data)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        path = self.path.split("?")[0]
        rt = STATE.rt
        if path == "/health":
            return _json(self, 200, {"ok": True, "status": "loaded" if rt.loaded else "preloading",
                                     "load_error": rt.load_error, "model": "LTX-2.3-distilled",
                                     "gpu": gpu_status(), **STATE.status()})
        parts = path.strip("/").split("/")
        if parts[:1] == ["jobs"] and len(parts) >= 2:
            job = STATE.get(parts[1]) if parts[1].isdigit() else None
            if job is None:
                return _json(self, 404, {"error": "no such job"})
            if len(parts) == 2:
                d = {"id": job.id, "status": job.status, "error": job.error, **job.meta}
                if job.mp4:
                    d["videos"] = [{"video_url": f"/jobs/{job.id}/videos/0",
                                    "keyframe_url": f"/jobs/{job.id}/videos/0/keyframe"}]
                return _json(self, 200, d)
            if parts[2:] == ["videos", "0"] and job.mp4:
                return _file(self, job.mp4, "video/mp4")
            # Canonical keyframe path matches the wan server + the app client
            # (/jobs/{id}/videos/0/keyframe). The app fetched that and got 404
            # ("not found", 22 bytes) here -> blank gallery posters. Keep the
            # old /jobs/{id}/keyframe alias for backward compatibility.
            if parts[2:] in (["videos", "0", "keyframe"], ["keyframe"]) and job.keyframe:
                return _file(self, job.keyframe, "image/png")
        return _json(self, 404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            body = self._body()
        except Exception as exc:
            return _json(self, 400, {"error": f"bad json: {exc}"})
        if path in ("/generate/enqueue", "/extend/enqueue"):
            if not STATE.rt.loaded:
                return _json(self, 503, {"error": "model still preloading"})
            job = STATE.enqueue(body)
            # Return BOTH "id" (the idea_rank client contract -- enqueue_video
            # parses .id, matching the Wan server) and "job_id" (older tools).
            return _json(self, 200, {"id": job.id, "job_id": job.id, "status": job.status})
        if path == "/embed":
            try:
                jid = body.get("job_id")
                if jid is not None:
                    job = STATE.get(jid)
                    if job is None or not job.mp4:
                        return _json(self, 400, {"error": "no ready video for job_id"})
                    mp4 = job.mp4
                else:
                    b64 = body.get("video_b64")
                    if not b64:
                        return _json(self, 400, {"error": "embed needs job_id or video_b64"})
                    mp4 = str(STATE.rt.out_dir / f"embed_{int(time.time()*1000)}.mp4")
                    Path(mp4).write_bytes(base64.b64decode(b64))
                return _json(self, 200, STATE.rt.embed_path(mp4))
            except Exception as exc:
                return _json(self, 500, {"error": f"{type(exc).__name__}: {exc}"})
        return _json(self, 404, {"error": "not found"})


def main():
    global STATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8910)
    ap.add_argument("--distilled", default=os.environ.get("LTX_DISTILLED", "/workspace/models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors"))
    ap.add_argument("--upsampler", default=os.environ.get("LTX_UPSAMPLER", "/workspace/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"))
    ap.add_argument("--gemma", default=os.environ.get("LTX_GEMMA", "/workspace/models/gemma-3-12b"))
    ap.add_argument("--offload", default=os.environ.get("LTX_OFFLOAD", "none"))
    ap.add_argument("--embed-model", default=os.environ.get("LTX_EMBED_MODEL", "MCG-NJU/videomae-base"))
    ap.add_argument("--out-dir", default=os.environ.get("LTX_OUT", "/workspace/ltx_out"))
    a = ap.parse_args()
    rt = Runtime(a.distilled, a.upsampler, a.gemma, a.offload, a.embed_model, a.out_dir)
    STATE = AppState(rt)
    def _pre():
        try:
            rt.preload()
        except Exception as exc:
            rt.load_error = f"{type(exc).__name__}: {exc}"
            print(f"preload failed: {rt.load_error}", flush=True)
    threading.Thread(target=_pre, name="ltx-preload", daemon=True).start()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"LTX server on {a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
