#!/usr/bin/env python3
"""GPU validation of ace-latent-pin-v1 + ace-latents-v1 on the real ACE-Step 1.5 XL SFT model.

Runs the server's own AceRuntime (engine v15, bf16, CUDA) on THIS directory's
ace_step_server.py (imported from next to this file, never /opt/media-servers)
and prints one JSON report (also written to OUT/report.json):

  1 receptive_field   decode a 20 s render's latents with a different tail after
                      frame K (and a different head before it): how far before
                      K*1920 (after it) the decoded audio changes (> 1e-6 etc.)
  2 decode_determinism  the same latents decoded twice, and decoded alone vs
                      inside the longer sequence (away from the edges)
  3 vae_round_trip    decode(encode(parent).mode()) vs the parent; encode(parent/g)
                      vs the parent's stored latents
  4 continuation      a 20 s text2music parent, then a stored-latents continuation
                      and a WAV-path continuation of the same parent (trim 5 s ->
                      C=375, G=250, J=362): job diagnostics, prefix bit-exactness,
                      context fidelity, sample step + spectral flux at the join and
                      at the cut vs the parent's own median/p99, levels
  5 timing_s          model load, each render, decodes

Run on spark-3 in a separate container from the spark-media image (the live
music renderer keeps running; this loads a second model copy, so claim the GPU
first: `resctl claim gpu-claim:spark-3 '45 minutes' ace-latent-pin-validation`):

  docker run --rm --gpus all --ulimit memlock=-1:-1 --ipc=host -e HF_HUB_OFFLINE=1 \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface:ro -v $PWD:/work -w /work \
    --entrypoint python idea-rank-spark-media:wan22-ace15 gpu_validate.py --out /work/gpu_validate_out

($PWD = a copy of this worktree on spark-3.) Exit status 1 if any section failed;
each failed section carries its error and traceback in the report.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("ACE_OUTPUT_RETENTION_MIN", "0")  # no pruner thread on the scratch dir
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

import ace_step_server as server  # noqa: E402

HOP = server.LATENT_HOP
RATE = server.ACE_SAMPLE_RATE


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clean(value):
    """JSON-safe: numpy scalars -> Python, inf/nan -> strings."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def corr(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None


def snr_db(reference, test):
    reference, test = np.asarray(reference, np.float64), np.asarray(test, np.float64)
    noise = float(np.sum((reference - test) ** 2))
    return math.inf if noise == 0 else 10 * math.log10(float(np.sum(reference ** 2)) / noise)


def compare16(reference, test):
    diff = test.astype(np.int64) - reference.astype(np.int64)
    return {"samples": int(diff.size), "identical_fraction": float(np.mean(diff == 0)),
            "max_abs_lsb": int(np.abs(diff).max()), "rms_lsb": float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))),
            "corr": corr(reference, test), "snr_db": snr_db(reference, test)}


def rms_db(x):
    value = float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))
    return 20 * math.log10(value) if value > 0 else -math.inf


def sample_steps(x):
    """|x[n] - x[n-1]|, max over channels; steps[0] = 0."""
    steps = np.zeros(len(x))
    steps[1:] = np.abs(np.diff(x, axis=0)).max(axis=1)
    return steps


def spectral_flux(mono, n_fft=2048, hop=512):
    """Half-wave rectified magnitude flux per STFT frame (Hann), with frame centres in samples."""
    frames = np.lib.stride_tricks.sliding_window_view(mono, n_fft)[::hop] * np.hanning(n_fft)
    magnitude = np.abs(np.fft.rfft(frames, axis=1))
    flux = np.zeros(len(magnitude))
    flux[1:] = np.maximum(magnitude[1:] - magnitude[:-1], 0).sum(axis=1)
    return flux, np.arange(len(magnitude)) * hop + n_fft // 2


def ratio(value, reference):
    """value / reference; None when the reference is 0 (e.g. digital silence)."""
    return value / reference if reference > 0 else None


def reference_stats(x):
    steps = sample_steps(x)[1:]
    flux = spectral_flux(x.mean(axis=1))[0][1:]
    return {"step_median": float(np.median(steps)), "step_p99": float(np.percentile(steps, 99)),
            "flux_median": float(np.median(flux)), "flux_p99": float(np.percentile(flux, 99))}


def event_metrics(x, position, reference):
    """Discontinuity measures at `position` (the first sample of the new side)."""
    steps = sample_steps(x)
    preceding = steps[max(1, position - 4800):position]  # the 100 ms before the event
    flux, centres = spectral_flux(x.mean(axis=1))
    near = flux[np.abs(centres - position) <= 1024]
    step, peak_flux = float(steps[position]), float(near.max())
    return {"sample": int(position), "step": step,
            "step_over_parent_median": ratio(step, reference["step_median"]),
            "step_over_parent_p99": ratio(step, reference["step_p99"]),
            "step_over_preceding_100ms_p99": ratio(step, float(np.percentile(preceding, 99))),
            "max_step_within_10ms": float(steps[position - 480:position + 480].max()),
            "flux_near": peak_flux,
            "flux_over_parent_median": ratio(peak_flux, reference["flux_median"]),
            "flux_over_parent_p99": ratio(peak_flux, reference["flux_p99"])}


def npy_b64(array):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def wav_b64(samples16):
    buffer = io.BytesIO()
    sf.write(buffer, samples16, RATE, format="WAV", subtype="PCM_16")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class Validation:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out).resolve()
        (self.out / "renders").mkdir(parents=True, exist_ok=True)
        self.timing = {}
        self.report = {"server_file": server.__file__, "args": vars(args), "timing_s": self.timing,
                       "versions": {"torch": torch.__version__, "numpy": np.__version__,
                                    "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__}}
        self.failed = []

    def timed(self, name, fn, *args, **kwargs):
        sync()
        start = time.time()
        value = fn(*args, **kwargs)
        sync()
        self.timing[name] = round(time.time() - start, 3)
        return value

    def section(self, name, fn):
        try:
            self.report[name] = fn()
        except Exception as exc:  # noqa: BLE001 -- recorded in the report and in the exit status
            self.report[name] = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            self.failed.append(name)

    def decode(self, latents):
        """[T, 64] latents -> float32 [T*1920, 2] through the server's own decode."""
        tensor = torch.as_tensor(np.ascontiguousarray(latents)).unsqueeze(0)
        return server._decode_latents(self.pipe, tensor.to(device=self.device, dtype=self.pipe.vae.dtype))

    def run(self):
        import diffusers

        self.report["versions"]["diffusers"] = diffusers.__version__
        self.runtime = server.AceRuntime(output_dir=self.out / "renders", checkpoint_dir="", dtype="bfloat16",
                                         engine="v15", model_id=self.args.model)
        self.pipe = self.timed("model_load", self.runtime._load)
        self.device = self.pipe._execution_device
        config = self.pipe.scheduler.config
        self.report["model"] = {
            "id": self.runtime._model_id, "revision": self.runtime.model_revision(), "device": str(self.device),
            "transformer_dtype": str(self.pipe.transformer.dtype), "vae_dtype": str(self.pipe.vae.dtype),
            "vae_use_tiling": bool(getattr(self.pipe.vae, "use_tiling", False)),
            "sample_rate": self.pipe.sample_rate, "latents_per_second": self.pipe.latents_per_second,
            "scheduler": {k: config.get(k) for k in ("num_train_timesteps", "shift", "use_dynamic_shifting",
                                                     "invert_sigmas", "stochastic_sampling", "shift_terminal")}}
        self.section("parent", self.render_parent)
        if "parent" in self.failed:
            return
        self.section("receptive_field", self.receptive_field)
        self.section("decode_determinism", self.decode_determinism)
        self.section("vae_round_trip", self.vae_round_trip)
        self.section("continuation", self.continuations)

    def render_parent(self):
        payload = {"prompt": self.args.prompt, "lyrics": "[instrumental]", "audio_duration": self.args.parent_seconds,
                   "infer_step": self.args.steps, "guidance_scale": 7.0, "shift": 3.0, "seed": self.args.seed}
        job = server.Job(id=1, payload=payload, seed=self.args.seed)
        self.timed("parent_text2music", self.runtime.generate, job)
        self.parent16 = sf.read(job.audios[0].path, dtype="int16", always_2d=True)[0]
        self.parent_latents = np.load(job.latents_path, allow_pickle=False)
        self.parent_meta = job.latents
        gain = float(job.latents["gain_envelope"][0][1])
        redecoded = self.timed("decode_parent", self.decode, self.parent_latents)
        again16 = server._pcm16(server._soft_clip(redecoded * np.float32(gain)))
        return {"frames": int(self.parent16.shape[0]), "latent_frames": int(self.parent_latents.shape[0]),
                "wav_is_latents_times_hop": self.parent16.shape[0] == self.parent_latents.shape[0] * HOP,
                "latents_metadata": job.latents, "peak_dbfs": rms_db([np.abs(self.parent16).max() / 32768]),
                "redecode_vs_stored_wav": compare16(self.parent16, again16)}

    def receptive_field(self):
        latents = self.parent_latents
        k = latents.shape[0] // 2
        base = self.decode(latents)
        tail, head = latents.copy(), latents.copy()
        tail[k:], head[:k] = latents[k:][::-1], latents[:k][::-1]  # realistic values, different content
        tail_diff = np.abs(self.timed("decode_tail_variant", self.decode, tail) - base).max(axis=1)
        head_diff = np.abs(self.decode(head) - base).max(axis=1)
        reach = {}
        for threshold in (1e-6, 1e-4, 1e-3, 1e-2):
            back = np.flatnonzero(tail_diff[:k * HOP] > threshold)
            forward = np.flatnonzero(head_diff[k * HOP:] > threshold)
            back_samples = int(k * HOP - back[0]) if back.size else 0
            forward_samples = int(forward[-1] + 1) if forward.size else 0
            reach[str(threshold)] = {"back_samples": back_samples, "back_frames": back_samples / HOP,
                                     "forward_samples": forward_samples, "forward_frames": forward_samples / HOP}
        return {"frames": int(latents.shape[0]), "k": k, "reach": reach,
                "run_to_run_max_abs_diff": float(np.abs(self.decode(latents) - base).max())}

    def decode_determinism(self):
        latents, (a, b), margin = self.parent_latents, (100, 300), 12
        whole = self.decode(latents)
        alone = self.decode(latents[a:b])
        inside = whole[a * HOP:b * HOP]
        interior = slice(margin * HOP, (b - a - margin) * HOP)
        diff = np.abs(alone[interior] - inside[interior])
        edge = np.abs(alone[:margin * HOP] - inside[:margin * HOP]).max(axis=1)
        return {"span_frames": [a, b], "margin_frames": margin,
                "run_to_run_max_abs_diff": float(np.abs(self.decode(latents) - whole).max()),
                "alone_vs_inside_interior_max_abs_diff": float(diff.max()),
                "alone_vs_inside_interior_max_lsb": float(diff.max() * 32768),
                "alone_vs_inside_interior_identical_fraction": float(np.mean(diff == 0)),
                "alone_vs_inside_start_edge_max_abs_diff_by_frame": [float(edge[f * HOP:(f + 1) * HOP].max())
                                                                     for f in range(margin)]}

    def encode(self, audio):
        """float [N, 2] -> posterior-mean latents [N/1920, 64] float32."""
        tensor = torch.from_numpy(np.ascontiguousarray(audio.T, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            mean = self.pipe.vae.encode(tensor.to(self.device, self.pipe.vae.dtype)).latent_dist.mode()
        return mean.transpose(1, 2)[0].float().cpu().numpy()

    def vae_round_trip(self):
        x = self.parent16.astype(np.float32) / 32768.0
        gain = np.float32(self.parent_meta["gain_envelope"][0][1])
        y = self.decode(self.timed("encode_parent", self.encode, x))
        interior = slice(RATE, len(x) - RATE)
        unscaled = self.encode(x / gain)
        stored = self.parent_latents
        return {"corr": corr(x, y), "snr_db": snr_db(x, y), "level_ratio_db": rms_db(y) - rms_db(x),
                "interior_corr": corr(x[interior], y[interior]), "interior_snr_db": snr_db(x[interior], y[interior]),
                "encode_of_audio_over_gain_vs_stored_latents": {
                    "corr": corr(unscaled, stored),
                    "relative_rms_error": float(np.sqrt(np.mean((unscaled - stored) ** 2)) / np.sqrt(np.mean(stored ** 2)))}}

    def continuations(self):
        trim = round(self.args.trim_seconds * RATE / HOP) * HOP  # the app's Contract B frame math
        retained = self.parent16.shape[0] - trim
        context_samples = min(retained, 30 * RATE) // HOP * HOP
        start = retained - context_samples
        context, generate = context_samples // HOP, self.args.generate_frames
        join = context - 13
        scaffold = np.zeros(((context + generate) * HOP, 2), dtype=np.int16)
        scaffold[:context_samples] = self.parent16[start:retained]
        spec = {"protocol": server.CONTINUATION_PROTOCOL, "context_frames": context, "generate_frames": generate,
                "join_frame": join, "crossfade_frames": 1, "latent_hop": HOP}
        base = {"prompt": self.args.prompt, "lyrics": "[instrumental]",
                "audio_duration": (context + generate) * HOP / RATE, "infer_step": self.args.steps,
                "guidance_scale": 7.0, "shift": 3.0, "seed": self.args.seed + 1, "audio_b64": wav_b64(scaffold)}
        parent_context = np.ascontiguousarray(self.parent_latents[start // HOP:start // HOP + context])
        requests = {
            "latents": {**base, "context_latents_b64": npy_b64(parent_context), "continuation": {
                **spec, "context_gain": [[s - start, g] for s, g in self.parent_meta["gain_envelope"]],
                "latents_model": self.parent_meta["model"], "latents_model_revision": self.parent_meta["model_revision"]}},
            "audio": {**base, "continuation": {**spec, "context_gain": None}}}
        reference = reference_stats(self.parent16.astype(np.float64) / 32768.0)
        result = {"geometry": {"context_frames": context, "generate_frames": generate, "join_frame": join,
                               "context_start_sample": start, "trim_samples": trim},
                  "parent_reference": reference}
        near_cut = context - server.DECODER_REACH_FRAMES
        for job_id, (name, payload) in enumerate(requests.items(), start=2):
            job = server.Job(id=job_id, payload=payload, seed=payload["seed"])
            self.timed(f"continuation_{name}", self.runtime.continuation, job)
            out = sf.read(job.audios[0].path, dtype="int16", always_2d=True)[0]
            child_latents = np.load(job.latents_path, allow_pickle=False)
            joined16 = np.concatenate([self.parent16[:start + join * HOP], out[join * HOP:]])
            sf.write(str(self.out / f"joined_{name}.wav"), joined16, RATE, subtype="PCM_16")
            joined = joined16.astype(np.float64) / 32768.0
            join_at, cut_at = start + join * HOP, start + context * HOP

            def span(a, b, out=out):
                return compare16(scaffold[a * HOP:b * HOP], out[a * HOP:b * HOP])

            result[name] = {
                "diagnostics": job.continuation, "latents_metadata": job.latents,
                "output_frames": int(out.shape[0]), "expected_frames": (context + generate) * HOP,
                "prefix_bit_exact": bool(np.array_equal(out[:join * HOP], scaffold[:join * HOP])),
                "crossfade_frame_vs_scaffold": span(join, join + 1),
                "context_vs_scaffold_crossfade_to_reach": span(join + 1, near_cut),  # outside the new latents' reach
                "context_vs_scaffold_last_10_frames": span(near_cut, context),
                "child_context_latents_vs_parent": {
                    "equal": bool(np.array_equal(child_latents[:context - join], parent_context[join:])),
                    "corr": corr(child_latents[:context - join], parent_context[join:])},
                "join": event_metrics(joined, join_at, reference),
                "cut": event_metrics(joined, cut_at, reference),
                "level_db": {"before_join_0.5s": rms_db(joined[join_at - RATE // 2:join_at]),
                             "after_join_0.5s": rms_db(joined[join_at:join_at + RATE // 2]),
                             "before_cut_2s": rms_db(joined[cut_at - 2 * RATE:cut_at]),
                             "after_cut_2s": rms_db(joined[cut_at:cut_at + 2 * RATE])}}
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "gpu_validate_out"))
    parser.add_argument("--model", default=server.DEFAULT_MODEL_V15)
    parser.add_argument("--prompt", default="deep house, warm rhodes chords, rolling bassline, crisp drums, 122 bpm")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--parent-seconds", type=float, default=20.0)
    parser.add_argument("--trim-seconds", type=float, default=5.0)
    parser.add_argument("--generate-frames", type=int, default=250)
    validation = Validation(parser.parse_args())
    try:
        with contextlib.redirect_stdout(sys.stderr):  # the server's progress lines; stdout = the report
            validation.run()
    except Exception as exc:  # noqa: BLE001 -- model load etc.: still report what ran
        validation.report["fatal"] = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        validation.failed.append("fatal")
    validation.report["failed_sections"] = validation.failed
    text = json.dumps(clean(validation.report), indent=2)
    (validation.out / "report.json").write_text(text)
    print(text, flush=True)
    return 1 if validation.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
