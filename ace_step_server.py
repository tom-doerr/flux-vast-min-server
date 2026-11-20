#!/usr/bin/env python3
"""ACE-Step music generation job server.

Mirrors wan_vast_min_server.py's job API (enqueue -> poll -> download) so the
idea_rank app client is a near-copy of the video client. Text tags + lyrics ->
a wav clip. Loads ACE-Step-v1-3.5B ONCE (lazy, lock-guarded), non-blocking
preload in a bg thread; a single worker thread drains the queue one job at a
time (one GPU). No silent fallbacks: a failed render marks the job 'error' with
the traceback rather than returning a bogus/empty clip.

One queue, three operations: text2music, repaint of marked `segments`, and a
`continuation` that pins the parent's latents (ace-latent-pin-v1, see
`AceRuntime.continuation`). v15 text2music and continuation also store their
final latents (`job_N.latents.npy`, GET /jobs/N/latents).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import queue
import re
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field, fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from render_receipts import OperationConflict, Receipts, file_identity

DEFAULT_MODEL = "ACE-Step/ACE-Step-v1-3.5B"
# ACE-Step 1.5 (diffusers AceStepPipeline): XL SFT = commercial-grade, stereo
# 48kHz, LM-planner-assisted long-range structure. Loaded via --engine v15.
DEFAULT_MODEL_V15 = "ACE-Step/acestep-v15-xl-sft-diffusers"
STATE: "AppState | None" = None


ACE_SAMPLE_RATE = 48000

# ACE-Step 1.5 VAE (AutoencoderOobleck, downsampling 2*4*4*6*10): one latent
# frame = 1920 samples = 40 ms; 64 channels.
LATENT_HOP = 1920
LATENT_CHANNELS = 64
# The pipeline's own output normalization: peak -> -1 dBFS.
TARGET_PEAK = 10.0 ** (-1.0 / 20.0)
# Memoryless soft clip on decode * gain: identity up to the knee, so a render
# normalized to -1 dBFS (0.891) passes through unchanged.
SOFT_CLIP_KNEE = 0.95
CONTINUATION_PROTOCOL = "ace-latent-pin-v1"
LATENTS_FORMAT = "ace-latents-v1"
CAPABILITIES = (f"continuation:{CONTINUATION_PROTOCOL}", f"latents:{LATENTS_FORMAT}")
MIN_CONTINUATION_FRAMES = 25  # 1 s of context and of new audio
# The decoder reaches ~9 frames around each latent: the join crossfade must end
# this many frames before the first new latent, so it blends only audio that
# decodes from the parent's context.
DECODER_REACH_FRAMES = 10
# A context re-encoded from audio decodes hot: decode(encode(x).mode()) came back
# 1.32 dB above the parent by least squares (corr 0.99, spark-3, Nov 20 2026), so
# without a fitted gain the join steps up in level. Below this RMS (-60 dBFS) the
# context is effectively silence and has no level to match.
FIT_MIN_RMS = 1e-3


def _normalize_repaint_spans(raw: Any) -> list[tuple[float, float]]:
    """Accept [{start,end}] or [[start,end]] and validate hard.

    The app merges and sorts spans before sending them (idea_rank's
    infill_segments), so anything overlapping or backwards here means a client
    bypassed that -- worth failing on rather than repainting a span twice.
    """
    spans: list[tuple[float, float]] = []
    for index, item in enumerate(raw or []):
        if isinstance(item, dict):
            start, end = item.get("start"), item.get("end")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            start, end = item
        else:
            raise ValueError(f"segment {index} must be {{start,end}} or [start,end]")
        start, end = float(start), float(end)
        if not (end > start >= 0):
            raise ValueError(f"segment {index} is not a forward span: {start}-{end}")
        spans.append((start, end))
    spans.sort()
    for (a_start, a_end), (b_start, _b_end) in zip(spans, spans[1:]):
        if b_start < a_end:
            raise ValueError(
                f"segments overlap ({a_start}-{a_end} and {b_start}-...); "
                "merge them before sending")
    return spans


# Equal-power crossfade at each splice boundary. Long enough to hide a phase
# discontinuity between source and regenerated audio, short enough not to eat
# into either side's content.
REPAINT_CROSSFADE_S = 0.020

# Extra audio repainted on EACH side of a marked span, as crossfade material.
#
# ★ DEFAULT 0 -- MEASURED WORSE, contrary to the obvious argument for it.
# The theory was that padding gives the fade model-audio on both sides instead
# of joining model to source cold. The theory is wrong about what the unpadded
# fade actually blends: repaint returns the SOURCE ROUND-TRIPPED through the VAE
# outside its window (measured corr 0.95 against the source), so the unpadded
# crossfade already blends source against near-source -- the ideal partner.
# Padding replaces that with newly generated content, which has no reason to
# match the source's level. Measured on the same track and spans:
#
#   seam   no pad              +250ms pad
#   5.0s   dip -0.2 dB         dip -2.8 dB
#   8.0s   dip -0.4 dB         dip -3.3 dB
#  20.0s   dip -0.0 dB         dip +2.0 dB
#
# Waveform continuity improved at one seam (step ratio 1.23 -> 0.07) and got
# worse at another, so there is no continuity win to trade the level loss for.
# Kept as a per-request knob (repaint_pad_s) because the result may differ for
# very short spans, where the model has less room to diverge.
REPAINT_PAD_S = 0.0


def _equal_power_ramp(length: int, dtype: Any, device: Any) -> tuple[Any, Any]:
    """Constant-power fade pair (sin/cos), NOT a linear one.

    ★ The two sides of a repaint seam are UNCORRELATED -- measured correlation
    inside a regenerated span is ~0.05 against the source. Linear (equal-gain)
    ramps sum to full amplitude only for correlated signals; for uncorrelated
    ones the power sums as g_a^2 + g_b^2, which at the midpoint of a linear
    fade is 0.5 -> a ~3 dB hole. Measured on a real splice before this fix:
    -1.4 to -3.2 dB dips at the seams. sin/cos keeps g_a^2 + g_b^2 == 1
    throughout, so the level holds.
    """
    import math

    import torch  # module-level torch is deliberately absent: it is a box-only dep

    theta = torch.linspace(0.0, math.pi / 2, length, dtype=dtype, device=device)
    return torch.sin(theta), torch.cos(theta)


def _splice_repaint(source: Any, repainted: Any, spans: list[tuple[float, float]],
                    sample_rate: int, crossfade_s: float = REPAINT_CROSSFADE_S) -> Any:
    """Keep the SOURCE audio outside the marked spans, bit for bit.

    ★ Why this exists: repaint round-trips the WHOLE track through the VAE, so
    the unmarked audio comes back as the same music at ~0.95 correlation rather
    than unchanged -- and iterating compounds that (0.95 after one pass, 0.89
    after two). Since the point of infill is to keep the good parts, the good
    parts are taken from the source and only the marked spans come from the
    model, with a short equal-power crossfade so the joins do not click.
    """
    import torch

    out = source.clone()
    total = out.shape[-1]
    # max(1, ...) would force a one-sample fade even when crossfading is off,
    # writing one sample PAST the span and breaking exact preservation.
    fade = max(0, int(round(float(crossfade_s) * int(sample_rate))))
    for start_s, end_s in spans:
        start = max(0, min(total, int(round(float(start_s) * sample_rate))))
        end = max(start, min(total, int(round(float(end_s) * sample_rate))))
        if end <= start:
            continue
        out[..., start:end] = repainted[..., start:end]
        # Fade the boundaries against the source we just cut into. Guarded so a
        # span at the very edge of the clip cannot read outside the buffer.
        head = min(fade, start, end - start)
        if head > 0:
            gain_in, gain_out = _equal_power_ramp(head, out.dtype, out.device)
            lo = start - head
            out[..., lo:start] = (source[..., lo:start] * gain_out
                                  + repainted[..., lo:start] * gain_in)
        tail = min(fade, total - end, end - start)
        if tail > 0:
            gain_in, gain_out = _equal_power_ramp(tail, out.dtype, out.device)
            out[..., end:end + tail] = (repainted[..., end:end + tail] * gain_out
                                        + source[..., end:end + tail] * gain_in)
    return out


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


def _soft_clip(audio: Any, knee: float = SOFT_CLIP_KNEE) -> Any:
    """y = x for |x| <= knee, else sign(x) * (k + (1-k) * tanh((|x|-k)/(1-k))).

    float32 and memoryless: re-applying it to the same decode reproduces the
    same samples, and |y| never exceeds 1 (no PCM16 wrap-around).
    """
    import numpy as np

    x = np.asarray(audio, dtype=np.float32)
    k = np.float32(knee)
    magnitude = np.abs(x)
    over = magnitude > k
    if not over.any():
        return x
    span = np.float32(1.0) - k
    soft = k + span * np.tanh((magnitude - k) / span)
    return np.where(over, np.sign(x) * soft, x).astype(np.float32)


def _gain_envelope(points: list[tuple[int, float]], length: int, cut: int) -> Any:
    """Per-sample float32 gain: piecewise linear between [sample, gain]
    breakpoints (constant outside them) before `cut`, the gain at `cut` after."""
    import numpy as np

    xs = np.array([p[0] for p in points], dtype=np.float64)
    gains = np.array([p[1] for p in points], dtype=np.float64)
    t = np.minimum(np.arange(length, dtype=np.float64), float(cut))
    return np.interp(t, xs, gains).astype(np.float32)


def _shift_envelope(points: list[tuple[int, float]], start: int, cut: int) -> list[list[float]]:
    """The same envelope in the coordinates of audio that begins at `start`:
    breakpoints inside (start, cut) move by -start, both ends are interpolated,
    and the gain at `cut` holds after it. A constant one comes back as [[0, g]]."""
    import numpy as np

    xs = [p[0] for p in points]
    gains = [p[1] for p in points]
    shifted = ([[0, float(np.interp(start, xs, gains))]]
               + [[s - start, g] for s, g in points if start < s < cut]
               + [[cut - start, float(np.interp(cut, xs, gains))]])
    while len(shifted) > 1 and shifted[-1][1] == shifted[-2][1]:
        shifted.pop()
    return shifted


def _pcm16(audio: Any) -> Any:
    """float [samples, channels] -> int16 exactly as sf.write(..., "PCM_16")
    converts it, so continuation audio quantizes like every stored render."""
    import numpy as np
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(audio, dtype=np.float32), ACE_SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    data, _sr = sf.read(buffer, dtype="int16", always_2d=True)
    return data


def _splice_continuation(scaffold: Any, rendered: Any, join_frame: int,
                         crossfade_frames: int, hop: int = LATENT_HOP) -> Any:
    """[0, join_frame*hop) = scaffold bit for bit; then a LINEAR crossfade to
    the render over crossfade_frames (weight i/n: its first sample is the
    scaffold's); then the render. int16 [samples, channels] in and out.

    Linear, not equal-power: both sides decode the same parent audio (corr
    ~1), and for correlated signals equal-gain weights keep the level.
    """
    import numpy as np

    if scaffold.shape != rendered.shape or scaffold.dtype != np.int16 or rendered.dtype != np.int16:
        raise ValueError(f"splice needs two equal int16 arrays, got {scaffold.dtype}{list(scaffold.shape)} "
                         f"and {rendered.dtype}{list(rendered.shape)}")
    start = join_frame * hop
    end = start + crossfade_frames * hop
    if join_frame < 0 or crossfade_frames < 1 or end > scaffold.shape[0]:
        raise ValueError(f"crossfade [{start}, {end}) does not fit {scaffold.shape[0]} samples")
    out = rendered.copy()
    out[:start] = scaffold[:start]
    weight = (np.arange(end - start, dtype=np.float64) / (end - start))[:, None]
    source = scaffold[start:end].astype(np.float64)
    out[start:end] = np.rint(source + weight * (rendered[start:end].astype(np.float64) - source)).astype(np.int16)
    return out


def _write_latents(path: Path, latents: Any) -> None:
    """[frames, 64] latents -> float32 .npy (bf16 -> fp32 is exact); frame k
    covers the stored audio's samples [k*1920, (k+1)*1920)."""
    import numpy as np

    array = latents.detach().float().cpu().numpy()
    if array.ndim != 2 or array.shape[1] != LATENT_CHANNELS:
        raise RuntimeError(f"latents must be [frames, {LATENT_CHANNELS}], got {list(array.shape)}")
    np.save(path, np.ascontiguousarray(array, dtype=np.float32), allow_pickle=False)


def _whole(spec: dict[str, Any], key: str) -> int:
    value = spec.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"continuation.{key} must be an integer, got {value!r}")
    return value


def _envelope_points(raw: Any) -> list[tuple[int, float]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("continuation.context_gain must be a non-empty list of [sample, gain]")
    points: list[tuple[int, float]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"context_gain[{index}] must be [sample, gain], got {item!r}")
        sample, gain = item
        if isinstance(sample, bool) or not isinstance(sample, int):
            raise ValueError(f"context_gain[{index}] sample must be an integer, got {sample!r}")
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or gain <= 0:
            raise ValueError(f"context_gain[{index}] gain must be a positive finite number, got {gain!r}")
        if points and sample <= points[-1][0]:
            raise ValueError(f"context_gain samples must increase strictly (index {index})")
        points.append((sample, float(gain)))
    return points


def _read_scaffold(audio_b64: Any, samples: int) -> Any:
    """The PCM16 48 kHz context + silence WAV -> int16 [samples, 2]."""
    import numpy as np
    import soundfile as sf

    if not audio_b64:
        raise ValueError("continuation requires audio_b64 (parent context + digital silence)")
    try:
        with sf.SoundFile(io.BytesIO(base64.b64decode(audio_b64, validate=True))) as wav:
            rate, channels, frames, subtype = wav.samplerate, wav.channels, wav.frames, wav.subtype
            data = wav.read(dtype="int16", always_2d=True)
    except sf.LibsndfileError as exc:
        raise ValueError(f"scaffold is not a readable WAV: {exc}") from exc
    if rate != ACE_SAMPLE_RATE or subtype != "PCM_16":
        raise ValueError(f"scaffold must be {ACE_SAMPLE_RATE} Hz PCM_16, got {rate} Hz {subtype}")
    if channels not in (1, 2):
        raise ValueError(f"scaffold has {channels} channels; only mono or stereo")
    if frames != samples or data.shape[0] != samples:
        raise ValueError(f"scaffold has {data.shape[0]} of {frames} frames; the request needs exactly "
                         f"(context_frames + generate_frames) * {LATENT_HOP} = {samples}")
    return np.repeat(data, 2, axis=1) if channels == 1 else data


def _read_context_latents(latents_b64: Any, frames: int) -> Any:
    import numpy as np

    try:
        array = np.load(io.BytesIO(base64.b64decode(latents_b64, validate=True)), allow_pickle=False)
    except (ValueError, OSError, EOFError) as exc:
        raise ValueError(f"context_latents_b64 is not a readable .npy: {exc}") from exc
    if array.dtype != np.float32 or array.shape != (frames, LATENT_CHANNELS):
        raise ValueError(f"context latents must be float32 [{frames}, {LATENT_CHANNELS}], "
                         f"got {array.dtype} {list(array.shape)}")
    if not np.isfinite(array).all():
        raise ValueError("context latents contain non-finite values")
    return array


@dataclass
class ContinuationRequest:
    context_frames: int
    generate_frames: int
    join_frame: int
    crossfade_frames: int
    context_gain: list[tuple[int, float]]
    scaffold: Any        # int16 [(C+G)*1920, 2]
    latents: Any = None  # float32 [C, 64], or None: encode the scaffold's context


def _continuation_request(payload: dict[str, Any], *, model: str, revision: str) -> ContinuationRequest:
    """Validate an ace-latent-pin-v1 request completely; never guess a value."""
    spec = payload.get("continuation")
    if not isinstance(spec, dict):
        raise ValueError("continuation must be an object")
    if spec.get("protocol") != CONTINUATION_PROTOCOL:
        raise ValueError(f"continuation protocol {spec.get('protocol')!r} is not {CONTINUATION_PROTOCOL!r}")
    context, generate, join, crossfade, hop = (_whole(spec, key) for key in (
        "context_frames", "generate_frames", "join_frame", "crossfade_frames", "latent_hop"))
    if hop != LATENT_HOP:
        raise ValueError(f"latent_hop {hop} is not this VAE's {LATENT_HOP}")
    if context < MIN_CONTINUATION_FRAMES or generate < MIN_CONTINUATION_FRAMES:
        raise ValueError(f"context_frames ({context}) and generate_frames ({generate}) must both be "
                         f">= {MIN_CONTINUATION_FRAMES} (1 s)")
    if crossfade < 1:
        raise ValueError(f"crossfade_frames must be >= 1, got {crossfade}")
    if join < 0 or join + crossfade > context - DECODER_REACH_FRAMES:
        raise ValueError(f"join_frame ({join}) + crossfade_frames ({crossfade}) must be <= context_frames "
                         f"({context}) - {DECODER_REACH_FRAMES}: the crossfade must stay out of the "
                         "decoder's reach of the new latents")
    duration = (context + generate) * LATENT_HOP / ACE_SAMPLE_RATE
    if payload.get("audio_duration") is not None and abs(float(payload["audio_duration"]) - duration) > 1e-6:
        raise ValueError(f"audio_duration {payload['audio_duration']} is not (C+G)*{LATENT_HOP}/"
                         f"{ACE_SAMPLE_RATE} = {duration}")
    gain_raw = spec.get("context_gain")
    source_model, source_revision = spec.get("latents_model"), spec.get("latents_model_revision")
    latents = None
    if payload.get("context_latents_b64") is not None:
        if gain_raw is None or not source_model or not source_revision:
            raise ValueError("context latents need continuation.context_gain, latents_model and "
                             "latents_model_revision")
        if (source_model, source_revision) != (model, revision):
            raise ValueError(f"context latents come from {source_model}@{source_revision}; this renderer "
                             f"runs {model}@{revision}")
        gain = _envelope_points(gain_raw)
        latents = _read_context_latents(payload["context_latents_b64"], context)
    elif gain_raw is not None or source_model or source_revision:
        raise ValueError("context_gain, latents_model and latents_model_revision belong to a request "
                         "that sends context_latents_b64")
    else:
        gain = [(0, 1.0)]  # re-encoded from the stored audio; continuation() fits the level after the decode
    scaffold = _read_scaffold(payload.get("audio_b64"), (context + generate) * LATENT_HOP)
    if scaffold[context * LATENT_HOP:].any():
        raise ValueError("scaffold must be digital silence after the context frames")
    return ContinuationRequest(context, generate, join, crossfade, gain, scaffold, latents)


def _operation(payload: dict[str, Any]) -> str:
    """repaint (segments), continuation, or generate (text2music)."""
    if "continuation" in payload:
        if "segments" in payload:
            raise ValueError("a request carries segments (repaint) or continuation, never both")
        return "continuation"
    return "repaint" if payload.get("segments") else "generate"


def _decode_latents(pipe: Any, latents: Any) -> Any:
    """[1, T, 64] latents -> float32 [T*1920, 2] audio in one untiled VAE pass.

    Untiled matters: a tiled decode depends on where the tiles fall, so a
    continuation's decode of its parent's latents would not match the parent.
    """
    import numpy as np
    import torch

    if pipe.sample_rate != ACE_SAMPLE_RATE or pipe.latents_per_second * LATENT_HOP != ACE_SAMPLE_RATE:
        raise RuntimeError(f"pipeline geometry {pipe.sample_rate} Hz / {pipe.latents_per_second} latents/s "
                           f"is not {ACE_SAMPLE_RATE} Hz / hop {LATENT_HOP}")
    if getattr(pipe.vae, "use_tiling", False):
        raise RuntimeError("VAE tiling is on; latents/continuation need one untiled decode")
    if latents.dim() != 3 or latents.shape[0] != 1 or latents.shape[2] != LATENT_CHANNELS:
        raise RuntimeError(f"expected latents [1, T, {LATENT_CHANNELS}], got {list(latents.shape)}")
    with torch.inference_mode():
        audio = pipe.vae.decode(latents.transpose(1, 2)).sample.float()
    expected = [1, 2, latents.shape[1] * LATENT_HOP]
    if list(audio.shape) != expected:
        raise RuntimeError(f"VAE decode returned {list(audio.shape)}, expected {expected}")
    decoded = np.ascontiguousarray(audio[0].cpu().numpy().T)
    if not np.isfinite(decoded).all():
        raise RuntimeError("VAE decode produced non-finite samples")
    return decoded


def _pin_schedule(pipe: Any, steps: int, shift: float, device: Any) -> tuple[list[float], list[float]]:
    """(t at each step, t the latents sit at after it): the pipeline's own
    schedule; after the final step t_next = 0 (the scheduler's terminal sigma)."""
    import torch

    schedule = pipe._get_timestep_schedule(num_inference_steps=steps, shift=shift, device=device,
                                           dtype=torch.float32).tolist()
    if len(schedule) != steps:
        raise RuntimeError(f"timestep schedule has {len(schedule)} entries for {steps} steps")
    return schedule, schedule[1:] + [0.0]


def _pinned_denoise(pipe: Any, *, context: Any, noise: Any, context_frames: int, generate_frames: int,
                    steps: int, shift: float, **call_kwargs: Any) -> Any:
    """ACE-Step repaint of frames [C, C+G) whose context frames follow the parent.

    After every Euler step the context frames are overwritten with the
    parent's latents noised to the step's new time, x_t = (1-t)*x0 + t*noise
    (rectified flow; the same noise as the initial latents), so at t=0 they ARE
    the parent's latents and the model continues the real parent rather than
    its own copy of it. For the call, an instance-level prepare_src_latents
    hands back the given context latents as the repaint conditioning instead
    of VAE-sampling src_audio. Returns the final [1, C+G, 64] latents.
    """
    import torch

    total = context_frames + generate_frames
    if list(noise.shape) != [1, total, LATENT_CHANNELS]:
        raise RuntimeError(f"noise must be [1, {total}, {LATENT_CHANNELS}], got {list(noise.shape)}")
    if list(context.shape) != [1, context_frames, LATENT_CHANNELS]:
        raise RuntimeError(f"context must be [1, {context_frames}, {LATENT_CHANNELS}], got {list(context.shape)}")
    # +0.25 frame keeps int(seconds * 25) on the intended frame despite float rounding.
    start, end = (context_frames + 0.25) / 25, (total + 0.25) / 25
    lps = pipe.latents_per_second
    if int(start * lps) != context_frames or int(end * lps) != total:
        raise RuntimeError(f"repaint window {start}-{end}s maps to frames {int(start * lps)}-{int(end * lps)}, "
                           f"not {context_frames}-{total}")
    schedule, t_next = _pin_schedule(pipe, steps, shift, noise.device)
    context32 = context.to(device=noise.device, dtype=torch.float32)
    noise32 = noise[:, :context_frames].float()
    src_full = torch.cat([context32, torch.zeros(1, generate_frames, LATENT_CHANNELS, device=noise.device)], dim=1)
    pinned_steps: list[int] = []

    def prepare_src_latents(*args: Any, **kwargs: Any) -> Any:
        if args:
            raise TypeError("prepare_src_latents called positionally; the pin wrapper only knows keywords")
        if kwargs.get("src_audio") is None:
            return original(**kwargs)  # the silence latent the pipeline tiles over the window
        batch = int(kwargs.get("batch_size") or 1)
        return src_full.to(device=kwargs["device"], dtype=kwargs["dtype"]).expand(batch, -1, -1), total

    def pin(_pipe: Any, step: int, timestep: Any, tensors: dict[str, Any]) -> dict[str, Any]:
        if abs(float(timestep) - schedule[step]) > 1e-6:
            raise RuntimeError(f"pipeline step {step} ran at t={float(timestep)}; the pin schedule says {schedule[step]}")
        latents = tensors["latents"]
        t = t_next[step]
        latents[:, :context_frames] = ((1.0 - t) * context32 + t * noise32).to(latents.dtype)
        pinned_steps.append(step)
        return {"latents": latents}

    own = "prepare_src_latents" in vars(pipe)
    original = pipe.prepare_src_latents
    pipe.prepare_src_latents = prepare_src_latents
    try:
        final = pipe(**call_kwargs, num_inference_steps=steps, shift=shift, task_type="repaint",
                     repainting_start=start, repainting_end=end, latents=noise, output_type="latent",
                     callback_on_step_end=pin, callback_on_step_end_tensor_inputs=["latents"]).audios
    finally:
        if own:
            pipe.prepare_src_latents = original
        else:
            del pipe.prepare_src_latents
    if pinned_steps != list(range(steps)):
        raise RuntimeError(f"the pin ran after steps {pinned_steps}, expected 0..{steps - 1}")
    if list(final.shape) != [1, total, LATENT_CHANNELS]:
        raise RuntimeError(f"pipeline returned latents {list(final.shape)}, expected [1, {total}, {LATENT_CHANNELS}]")
    if not torch.equal(final[:, :context_frames], context32.to(final.dtype)):
        raise RuntimeError("final context latents differ from the pinned parent latents")
    return final


def _fit_context_gain(scaffold: Any, decoded: Any, context_frames: int) -> tuple[float, dict[str, Any] | None]:
    """Least-squares gain that maps the decode of a re-encoded context onto the
    scaffold, fitted where neither window edge nor the new latents reach.
    Returns (gain, fit diagnostics); (1.0, None) for an effectively silent context."""
    import numpy as np

    lo, hi = DECODER_REACH_FRAMES * LATENT_HOP, (context_frames - DECODER_REACH_FRAMES) * LATENT_HOP
    reference = scaffold[lo:hi].astype(np.float64) / 32768.0
    decode = decoded[lo:hi].astype(np.float64)
    if float(np.sqrt(np.mean(reference ** 2))) < FIT_MIN_RMS:
        return 1.0, None
    energy = float((decode * decode).sum())
    gain = float((reference * decode).sum()) / energy if energy > 0 else 0.0
    if not 0.25 <= gain <= 4.0:
        raise RuntimeError(f"the re-encoded context fits gain {gain:.4f}; its decode does not reproduce "
                           "the parent's audio")
    fitted = (gain * decode).ravel()
    correlation = float(np.corrcoef(reference.ravel(), fitted)[0, 1]) if fitted.std() > 0 else None
    return gain, {"gain": gain, "gain_db": 20 * math.log10(gain), "correlation": correlation,
                  "frames": [lo // LATENT_HOP, hi // LATENT_HOP]}


def _continuation_diagnostics(source: str, request: ContinuationRequest, rendered: Any,
                              unclipped: Any) -> dict[str, Any]:
    """Join frame: scaffold vs the (soft-clipped) render there. new_peak_before_clip
    covers the new frames [C, C+G); soft_clipped_samples counts channel samples
    over the child's span [J*1920, end) that the clip changed."""
    import numpy as np

    start = request.join_frame * LATENT_HOP
    end = start + request.crossfade_frames * LATENT_HOP
    scaffold = request.scaffold[start:end].astype(np.float64) / 32768.0
    render = rendered[start:end].astype(np.float64)
    correlation = None  # undefined when either side is constant (e.g. digital silence)
    if scaffold.std() > 0 and render.std() > 0:
        correlation = float(np.corrcoef(scaffold.ravel(), render.ravel())[0, 1])
    knee = np.float32(SOFT_CLIP_KNEE)
    return {"context_source": source, "join_sample": start, "crossfade_samples": end - start,
            "join_max_abs_diff": float(np.abs(scaffold - render).max()), "join_correlation": correlation,
            "new_peak_before_clip": float(np.abs(unclipped[request.context_frames * LATENT_HOP:]).max()),
            "soft_clipped_samples": int((np.abs(unclipped[start:]) > knee).sum())}


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
    operation_id: str = ""
    request_sha256: str = ""
    seed: int | None = None
    retain_outputs: bool = True
    renderer_identity: dict[str, Any] = field(default_factory=dict)
    recipe_sha256: str = ""
    output_identity: dict[str, Any] = field(default_factory=dict)
    latents: dict[str, Any] = field(default_factory=dict)       # ace-latents-v1 metadata
    latents_path: str = ""
    continuation: dict[str, Any] = field(default_factory=dict)  # seam diagnostics

    @classmethod
    def restore(cls, record: dict[str, Any]) -> "Job":
        values = {key: value for key, value in record.items() if key in {f.name for f in fields(cls)}}
        values["audios"] = [AudioRecord(Path(a["path"]), a["duration"], a["sample_rate"],
            Path(a["waveform_path"]) if a.get("waveform_path") else None) for a in values.get("audios", [])]
        return cls(**values)

    def payload_dict(self) -> dict[str, Any]:
        expired = (any(not a.path.is_file() for a in self.audios)
                   or bool(self.latents_path and not Path(self.latents_path).is_file()))
        data = {
            "id": self.id,
            "operation_id": self.operation_id,
            "request_sha256": self.request_sha256,
            "seed": self.seed,
            "output_identity": self.output_identity,
            "status": "output_expired" if self.status == "done" and expired else self.status,
            "error": self.error,
            "audios": [a.payload(self.id, i) for i, a in enumerate(self.audios)],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if self.latents:
            data["latents"] = self.latents
        if self.continuation:
            data["continuation"] = self.continuation
        return data


def _pipeline_seed(payload: dict[str, Any]) -> dict[str, Any]:
    # Preserve the original pipeline defaults when no explicit seed was sent.
    if payload.get("seed") is None:
        return {}
    import torch
    return {"generator": torch.Generator(device="cuda").manual_seed(int(payload["seed"]))}


def _snapshot_revision(model_id: str) -> str:
    """The commit sha (first 16 chars) of model_id's snapshot in the local HF
    cache; raises when the cache has none. No network."""
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(model_id, local_files_only=True))
    if path.parent.name != "snapshots" or not re.fullmatch(r"[0-9a-f]{40}", path.name):
        raise RuntimeError(f"{path} is not an HF cache snapshot folder")
    return path.name[:16]


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
        self._model_revision_cache = ""  # a found revision, looked up with the model loaded
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
                    retained = self.receipts.retained_ids() if getattr(self, "receipts", None) else None
                    for f in self.output_dir.glob("*"):
                        # Until AppState has opened the journal, do not prune
                        # any job output. Later, only acknowledged outputs age out.
                        token = f.stem.split(".", 1)[0].removeprefix("job_")
                        if f.name.startswith("job_") and (retained is None or (token.isdigit() and int(token) in retained)):
                            continue
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
        model hash for per-item provenance; "" while it cannot be found. Only a
        found revision, looked up with the model loaded, is cached: /health
        polled during the preload can look before the load has downloaded its
        snapshot (a miss, or an older snapshot), and a cached miss would disable
        every v15 latents job until a restart."""
        if self._model_revision_cache:
            return self._model_revision_cache
        loaded = self._pipe is not None
        try:
            revision = _snapshot_revision(self._model_id)
        except Exception:  # noqa: BLE001 -- "" = unknown here; _latents_revision raises the cause
            return ""
        if loaded:
            self._model_revision_cache = revision
        return revision

    def _latents_revision(self) -> str:
        """The revision latents are labelled with: they are only reusable by the
        exact model that made them. Called with the model loaded; raises when
        the revision cannot be found (the next call looks again)."""
        if not self._model_revision_cache:
            try:
                self._model_revision_cache = _snapshot_revision(self._model_id)
            except Exception as exc:  # noqa: BLE001 -- re-raised with the model and the cause
                raise RuntimeError(f"no HF snapshot revision found for {self._model_id} ({type(exc).__name__}: "
                                   f"{exc}); latents cannot be labelled with their model") from exc
        return self._model_revision_cache

    def _store_latents(self, job: Job, latents: Any, *, revision: str, gain_envelope: list[list[float]],
                       render_start_frame: int) -> None:
        """Write job_N.latents.npy (pruned with the job's other files) and its
        ace-latents-v1 metadata: decode(latents) * gain_envelope, soft-clipped
        at the knee, is the stored audio (sample 0 = latent frame 0). revision =
        the caller's _latents_revision(), resolved before its render."""
        path = self.output_dir / f"job_{job.id:06d}.latents.npy"
        _write_latents(path, latents)
        job.latents_path = str(path)
        job.latents = {"format": LATENTS_FORMAT, "url": f"/jobs/{job.id}/latents",
                       "frames": int(latents.shape[0]), "latent_hop": LATENT_HOP,
                       "sample_rate": ACE_SAMPLE_RATE, "channels": LATENT_CHANNELS,
                       "gain_envelope": gain_envelope, "soft_clip_knee": SOFT_CLIP_KNEE,
                       "model": self._model_id, "model_revision": revision,
                       "render_start_frame": render_start_frame}

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
                # Every v15 render labels its latents with the revision: look it up now
                # that the load has its snapshot, so the label names what was loaded and
                # a missing one fails the preload here, not each later job. The pipeline
                # stays loaded: repaint needs no revision, and generate/continuation
                # look again before they render.
                self._model_revision_cache = ""
                print(f"model revision {self._latents_revision()}", flush=True)
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

    def _load_src_audio(self, path: Path) -> Any:
        """Load a wav as the [channels, samples] 48 kHz tensor repaint wants.

        ACE-Step v1.5 documents src_audio as "[channels, samples] at 48kHz";
        soundfile hands back [samples, channels], and any clip that was written
        at a different rate has to be resampled or the repaint window lands on
        the wrong part of the track.
        """
        import numpy as np
        import soundfile as sf
        import torch

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        audio = torch.from_numpy(np.ascontiguousarray(data.T))  # -> [channels, samples]
        if int(sr) != ACE_SAMPLE_RATE:
            import torchaudio

            audio = torchaudio.functional.resample(audio, int(sr), ACE_SAMPLE_RATE)
        return audio.contiguous()

    def repaint(self, job: Job) -> None:
        """Regenerate ONLY the marked spans of an existing track.

        The pipeline repaints one window per call, so N spans are N sequential
        passes, each fed the previous pass's audio. Spans arrive already merged
        and sorted (the app's infill_segments owns that), so passes never
        overlap and the pass count is exactly len(spans).
        """
        if self.engine != "v15":
            raise RuntimeError(
                "repaint requires --engine v15; the v1 acestep pipeline has no "
                "repainting_start/repainting_end parameters")
        import soundfile as sf

        pipe = self._load()
        p = job.payload or {}
        spans = _normalize_repaint_spans(p.get("segments"))
        if not spans:
            raise ValueError("repaint requires at least one {start,end} segment")
        src_path, is_temp = self._resolve_audio_path(p)
        try:
            audio = self._load_src_audio(src_path)
        finally:
            if is_temp:
                src_path.unlink(missing_ok=True)
        duration = audio.shape[-1] / float(ACE_SAMPLE_RATE)
        prompt = str(p.get("prompt") or "")
        lyrics = str(p.get("lyrics") or "[instrumental]")
        g = float(p.get("guidance_scale", 7.0))
        if g > 10.0:
            g = 7.0
        steps = max(1, int(p.get("infer_step", 40)))
        seeded = _pipeline_seed(p)
        save_path = Path(self.output_dir / f"job_{job.id:06d}.wav")
        t0 = time.time()
        pad = max(0.0, float(p.get("repaint_pad_s", REPAINT_PAD_S)))
        if pad and pad <= REPAINT_CROSSFADE_S:
            raise ValueError(
                f"repaint_pad_s ({pad}) must exceed the crossfade "
                f"({REPAINT_CROSSFADE_S}) or there is nothing to fade within")
        for index, (start, end) in enumerate(spans):
            if start >= duration:
                raise ValueError(
                    f"segment {index} starts at {start:.3f}s, past the "
                    f"{duration:.3f}s track")
            # Repaint WIDER than asked, then splice at the requested edges: the
            # crossfade then blends model->model (both conditioned on the same
            # source) instead of model->source, which is where a cold join
            # audibly steps.
            gen_start = max(0.0, start - pad)
            gen_end = min(duration, end + pad)
            with self._lock:
                out = pipe(
                    prompt=prompt,
                    lyrics=lyrics,
                    audio_duration=duration,
                    num_inference_steps=steps,
                    guidance_scale=g,
                    shift=float(p.get("shift", 3.0)),
                    **seeded,
                    task_type="repaint",
                    src_audio=audio,
                    repainting_start=float(gen_start),
                    repainting_end=float(gen_end),
                ).audios
            produced = out[0].detach().cpu().float()
            if produced.dim() == 1:
                produced = produced.unsqueeze(0)
            if produced.shape[-1] != audio.shape[-1]:
                raise RuntimeError(
                    f"repaint returned {produced.shape[-1]} samples for a "
                    f"{audio.shape[-1]}-sample track; cannot splice")
            # Splice per pass, so the NEXT pass reads source-quality audio
            # everywhere except the spans already repainted.
            audio = _splice_repaint(
                audio, produced, [(start, min(end, duration))], ACE_SAMPLE_RATE)
            print(f"[ace] repaint job {job.id} pass {index + 1}/{len(spans)} "
                  f"{start:.2f}-{end:.2f}s (generated {gen_start:.2f}-"
                  f"{gen_end:.2f}s, pad {pad:.2f}s) spliced", flush=True)
        sf.write(str(save_path), audio.T.numpy(), ACE_SAMPLE_RATE)
        _to_pcm16(save_path)
        dur, sr = _wav_info(save_path)
        rec = AudioRecord(save_path, dur, sr)
        waveform = save_path.with_name(save_path.stem + ".waveform.png")
        try:
            _waveform_png(save_path, waveform)
            rec.waveform_path = waveform
        except Exception as exc:  # noqa: BLE001 -- visual is non-fatal
            print(f"[ace] waveform failed job {job.id}: {exc}", flush=True)
        job.audios = [rec]
        print(f"[ace] repaint job {job.id} -> {save_path.name} {len(spans)} pass(es) "
              f"{time.time() - t0:.1f}s", flush=True)

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
            # The latents' label, before the render: a missing one costs no GPU
            # time and leaves no WAV behind (an error receipt is never acked, so
            # the pruner would keep it forever).
            revision = self._latents_revision()
            # v1.5 defaults: SFT wants ~30-60 steps, guidance 7 (APG), shift 3.
            # A v1-era guidance (>10) from an un-updated client is reset, since
            # 15 wrecks v1.5 output. Output is stereo 48kHz float.
            g = float(p.get("guidance_scale", 7.0))
            if g > 10.0:
                g = 7.0
            steps = int(p.get("infer_step", 40))
            with self._lock:
                # Latents out, decoded here: they are kept for continuations,
                # together with the exact gain that turned them into the WAV.
                latents = pipe(
                    prompt=prompt,
                    lyrics=lyrics,
                    audio_duration=float(p.get("audio_duration", 30.0)),
                    num_inference_steps=max(1, steps),
                    guidance_scale=g,
                    shift=float(p.get("shift", 3.0)),
                    output_type="latent",
                    **_pipeline_seed(p),
                ).audios
                decoded = _decode_latents(pipe, latents)
            wav = Path(save_path)
            self._write_text2music(job, wav, latents, decoded, revision=revision)
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
                    manual_seeds=p.get("manual_seeds") if p.get("manual_seeds") is not None else (str(int(p["seed"])) if p.get("seed") is not None else None),
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

    def _write_text2music(self, job: Job, wav: Path, latents: Any, decoded: Any, *, revision: str) -> None:
        """The pipeline's own output level (its two normalization stages are one
        gain, TARGET_PEAK / peak), written as PCM16 next to the latents."""
        import numpy as np
        import soundfile as sf

        gain = TARGET_PEAK / max(float(np.abs(decoded).max()), 1e-6)
        sf.write(str(wav), _soft_clip(decoded * np.float32(gain)), ACE_SAMPLE_RATE, subtype="PCM_16")
        self._store_latents(job, latents[0], revision=revision, gain_envelope=[[0, gain]], render_start_frame=0)

    def continuation(self, job: Job) -> None:
        """Continue a track from its real audio (ace-latent-pin-v1).

        The scaffold is the parent's last C latent frames of audio plus G frames
        of digital silence. Its C context frames stay pinned to the parent's
        latents (sent with the request, else the posterior mean of the
        scaffold's context) while the model writes the G new frames, so the
        decode continues the parent itself. Output, (C+G)*1920 samples:
        [0, J*1920) = the scaffold bit for bit, frame J a linear crossfade to
        the decode, then soft_clip(decode * context gain). The child's latents
        are frames [J, C+G); the app stores the child from sample J*1920.
        """
        if self.engine != "v15":
            raise RuntimeError("continuation requires --engine v15 (the diffusers AceStepPipeline)")
        import numpy as np
        import soundfile as sf
        import torch

        p = job.payload or {}
        # Load first: the context latents must match the revision that was loaded,
        # not one a lookup saw before the load downloaded the model.
        pipe = self._load()
        revision = self._latents_revision()
        request = _continuation_request(p, model=self._model_id, revision=revision)
        context, generate = request.context_frames, request.generate_frames
        total = context + generate
        g = float(p.get("guidance_scale", 7.0))
        if g > 10.0:
            g = 7.0
        scaffold = torch.from_numpy(np.ascontiguousarray(request.scaffold.T, dtype=np.float32) / 32768.0)
        save_path = self.output_dir / f"job_{job.id:06d}.wav"
        t0 = time.time()
        with self._lock:
            device = pipe._execution_device
            if request.latents is not None:
                context_latents = torch.from_numpy(request.latents).unsqueeze(0).to(device)
            else:
                with torch.no_grad():  # posterior MEAN; the pipeline itself would sample()
                    encoded = pipe.vae.encode(scaffold[:, :context * LATENT_HOP].unsqueeze(0).to(
                        device=device, dtype=pipe.vae.dtype)).latent_dist.mode()
                if list(encoded.shape) != [1, LATENT_CHANNELS, context]:
                    raise RuntimeError(f"VAE encode returned {list(encoded.shape)}, "
                                       f"expected [1, {LATENT_CHANNELS}, {context}]")
                context_latents = encoded.transpose(1, 2).float()
            # torch.randn with one generator on the pipeline's device = diffusers' randn_tensor
            generator = (torch.Generator(device=device).manual_seed(int(p["seed"]))
                         if p.get("seed") is not None else None)
            noise = torch.randn((1, total, LATENT_CHANNELS), generator=generator, device=device,
                                dtype=pipe.transformer.dtype)
            final = _pinned_denoise(
                pipe, context=context_latents, noise=noise, context_frames=context, generate_frames=generate,
                steps=max(1, int(p.get("infer_step", 40))), shift=float(p.get("shift", 3.0)),
                prompt=str(p.get("prompt") or ""), lyrics=str(p.get("lyrics") or "[instrumental]"),
                audio_duration=total * LATENT_HOP / ACE_SAMPLE_RATE, guidance_scale=g, src_audio=scaffold)
            decoded = _decode_latents(pipe, final)
        cut = context * LATENT_HOP
        context_gain, gain_fit = request.context_gain, None
        if request.latents is None:  # stored latents carry their exact gain; a re-encode needs a fit
            fitted, gain_fit = _fit_context_gain(request.scaffold, decoded, context)
            context_gain = [(0, fitted)]
        unclipped = decoded * _gain_envelope(context_gain, total * LATENT_HOP, cut)[:, None]
        if not np.isfinite(unclipped).all():
            raise ValueError("decode * context_gain overflows float32")
        rendered = _soft_clip(unclipped)
        out = _splice_continuation(request.scaffold, _pcm16(rendered), request.join_frame, request.crossfade_frames)
        sf.write(str(save_path), out, ACE_SAMPLE_RATE, subtype="PCM_16")
        join = request.join_frame * LATENT_HOP
        self._store_latents(job, final[0, request.join_frame:], revision=revision,
                            gain_envelope=_shift_envelope(context_gain, join, cut),
                            render_start_frame=request.join_frame)
        source = "latents" if request.latents is not None else "audio"
        job.continuation = {**_continuation_diagnostics(source, request, rendered, unclipped),
                            "context_gain_fit": gain_fit}
        dur, sr = _wav_info(save_path)
        rec = AudioRecord(save_path, dur, sr)
        waveform = save_path.with_name(save_path.stem + ".waveform.png")
        try:
            _waveform_png(save_path, waveform)
            rec.waveform_path = waveform
        except Exception as exc:  # noqa: BLE001 -- visual is non-fatal
            print(f"[ace] waveform failed job {job.id}: {exc}", flush=True)
        job.audios = [rec]
        print(f"[ace] continuation job {job.id} -> {save_path.name} C={context} G={generate} "
              f"J={request.join_frame} context={source} join_corr={job.continuation['join_correlation']} "
              f"{time.time() - t0:.1f}s", flush=True)


class AppState:
    def __init__(self, runtime: AceRuntime) -> None:
        self.runtime = runtime
        self._jobs: dict[int, Job] = {}
        self.receipts = Receipts(runtime.output_dir / ".render-receipts", exclusive=True)
        runtime.receipts = self.receipts
        self._lock = threading.Lock()
        self._queue: "queue.Queue[int]" = queue.Queue()
        for record in self.receipts.all():
            job = Job.restore(record)
            self._jobs[job.id] = job
            if job.status == "running":
                job.status = "interrupted"
                job.error = "Renderer interrupted before completed output manifest; retained without resubmission"
                self.receipts.save(asdict(job))
            elif job.status == "queued":
                self._queue.put(job.id)
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
            if payload.get("operation_id"):
                previous = self.receipts.find(payload["operation_id"], payload)
                if previous:
                    return self._jobs[previous["id"]]
            _operation(payload)  # segments + continuation together is refused here
            seed = int(payload["seed"]) if payload.get("seed") is not None else None
            if seed is not None and not 0 <= seed < 2**63:
                raise ValueError("seed must be a nonnegative 63-bit integer")
            if seed is not None and payload.get("manual_seeds") is not None and str(payload["manual_seeds"]) != str(seed):
                raise ValueError("manual_seeds conflicts with the frozen seed")
            frozen = Job(id=0, payload=dict(payload), seed=seed, retain_outputs=bool(payload.get("operation_id")),
                         renderer_identity=self._identity(),
                         recipe_sha256=file_identity(__file__)["sha256"])
            job = Job.restore(self.receipts.create(payload, asdict(frozen)))
            self._jobs[job.id] = job
        self._queue.put(job.id)
        return job

    def _identity(self) -> dict[str, Any]:
        return {"model": self.runtime._model_id, "engine": self.runtime.engine,
                "dtype": self.runtime.dtype, "checkpoint_dir": self.runtime.checkpoint_dir}

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
                "queue_size": pending, "model": self.runtime._model_id, "operation_protocol": "durable-v1"}

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None:
                continue
            job.status = "running"
            job.started_at = time.time()
            try:
                if job.renderer_identity != self._identity() or job.recipe_sha256 != file_identity(__file__)["sha256"]:
                    raise RuntimeError("Queued operation belongs to a different renderer model or recipe")
                self.receipts.save(asdict(job))
                # One queue, three operations: segments = infill of an existing
                # track, continuation = extend one, everything else = text2music.
                operation = _operation(job.payload or {})
                if operation == "repaint":
                    self.runtime.repaint(job)
                elif operation == "continuation":
                    self.runtime.continuation(job)
                else:
                    self.runtime.generate(job)
                if len(job.audios) != 1:
                    raise RuntimeError("Expected exactly one completed audio output")
                job.output_identity = {"wav": file_identity(job.audios[0].path)}
                if job.audios[0].waveform_path:
                    job.output_identity["waveform"] = file_identity(job.audios[0].waveform_path)
                if job.latents_path:
                    job.output_identity["latents"] = file_identity(job.latents_path)
                job.status = "done"
            except Exception as exc:  # noqa: BLE001
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.traceback = traceback.format_exc()
                print(f"[ace] job {job_id} FAILED: {job.error}\n{job.traceback}", flush=True)
            finally:
                job.finished_at = time.time()
                self.receipts.save(asdict(job))
                self._queue.task_done()

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
                               # A client must see these before it sends a continuation:
                               # an older server would render that payload as text2music.
                               "capabilities": list(CAPABILITIES) if rt.engine == "v15" else [],
                               **self.state.status(), "gpu": gpu_status()})
        parts = path.strip("/").split("/")
        if len(parts) == 2 and parts[0] == "operations":
            try:
                record = self.state.receipts.find(parts[1])
            except ValueError as exc:
                return self._json({"error": str(exc)}, 400)
            return self._json(self.state.get(record["id"]).payload_dict()) if record else self._json({"error": "not found"}, 404)
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
        if len(parts) == 3 and parts[2] == "latents":
            if not job.latents_path or not Path(job.latents_path).is_file():
                return self._json({"error": "latents not available"}, 404)
            return self._bytes(Path(job.latents_path).read_bytes(), "application/octet-stream")
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
        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "operations" and parts[2] == "ack":
            try:
                record = self.state.receipts.acknowledge(parts[1], str(payload.get("request_sha256") or ""))
            except ValueError as exc:
                return self._json({"error": str(exc)}, 409)
            return self._json({"ok": True, "operation_id": record["operation_id"]})
        if path == "/generate/enqueue":
            try:
                job = self.state.enqueue(payload)
            except OperationConflict as exc:
                return self._json({"error": str(exc)}, 409)
            except ValueError as exc:
                return self._json({"error": str(exc)}, 400)
            return self._json(job.payload_dict())
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
