"""ace-latent-pin-v1 continuation + latents artifact on CPU: real torch/numpy/soundfile,
a fake AceStepPipeline with the real call/callback contract, no model or network."""
import base64
import functools
import io
import math
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

import ace_step_server as server

HOP = server.LATENT_HOP
MODEL, REVISION = "ACE-Step/test-model", "0123456789abcdef"
SNAPSHOT = REVISION + "0" * 24  # the cache's commit folder; the label is its first 16 chars


class FakeVae:
    """Oobleck-shaped: encode = per-frame channel means spread over 64 latent channels
    (posterior mean = mode), decode = latent channels 0/1 held for one hop."""
    dtype = torch.bfloat16
    use_tiling = False

    def encode(self, audio):
        frames = audio.shape[-1] // HOP
        mean = audio.float().reshape(1, 2, frames, HOP).mean(-1).repeat(1, 32, 1).to(self.dtype)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: mean, sample=lambda: mean + 1))

    def decode(self, latents):
        return SimpleNamespace(sample=latents[:, :2].repeat_interleave(HOP, dim=-1))


class FakePipe:
    """What AceStepPipeline.__call__ does that the pin relies on, in its order:
    prepare_src_latents(src_audio) sets the latent length, the repaint window comes from
    int(seconds * 25), the silence latent replaces the window, Euler steps over
    _get_timestep_schedule plus the terminal 0, callback_on_step_end after every step."""
    sample_rate = 48000
    latents_per_second = 25.0
    _execution_device = torch.device("cpu")

    def __init__(self, fail_at_step=None, call_callback=True):
        self.vae = FakeVae()
        self.transformer = SimpleNamespace(dtype=torch.bfloat16, config=SimpleNamespace(audio_acoustic_hidden_dim=64))
        self.fail_at_step, self.call_callback = fail_at_step, call_callback
        self.calls, self.trace, self.conditioning = [], [], None

    def _get_timestep_schedule(self, num_inference_steps=8, shift=3.0, device=None, dtype=None, timesteps=None):
        t = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=dtype)
        if shift != 1.0:
            t = shift * t / (1 + (shift - 1) * t)
        return t[:-1]

    def prepare_src_latents(self, device, dtype, batch_size=1, src_audio=None, audio_codes=None,
                            latent_length=None, task_type="text2music"):
        if src_audio is not None:
            raise AssertionError("the pipeline must never VAE-sample the continuation scaffold")
        silence = torch.full((1, latent_length, 64), -0.5, device=device, dtype=dtype)
        return silence.expand(batch_size, -1, -1), latent_length

    def __call__(self, prompt, lyrics="", audio_duration=60.0, num_inference_steps=8, guidance_scale=7.0,
                 shift=3.0, generator=None, latents=None, output_type="pt", callback_on_step_end=None,
                 callback_on_step_end_tensor_inputs=("latents",), task_type="text2music", src_audio=None,
                 repainting_start=None, repainting_end=None):
        self.calls.append(dict(prompt=prompt, latents=latents, output_type=output_type, task_type=task_type,
                               repainting_start=repainting_start, repainting_end=repainting_end,
                               tensor_inputs=list(callback_on_step_end_tensor_inputs), src_audio=src_audio))
        device, dtype = self._execution_device, self.transformer.dtype
        length = math.ceil(audio_duration * self.latents_per_second)
        if src_audio is not None:
            src, length = self.prepare_src_latents(device=device, dtype=dtype, batch_size=1, src_audio=src_audio,
                                                   audio_codes=None, latent_length=length, task_type=task_type)
            mask = torch.zeros(1, length, 64, device=device, dtype=dtype)
            mask[:, int(repainting_start * 25):int(repainting_end * 25)] = 1.0
            silence, _ = self.prepare_src_latents(device=device, dtype=dtype, batch_size=1, latent_length=length)
            self.conditioning = torch.where(mask > 0.5, silence, src)
        if latents is None:
            latents = torch.randn((1, length, 64), generator=generator, device=device, dtype=dtype)
        xt = latents.to(device=device, dtype=dtype)
        sigmas = self._get_timestep_schedule(num_inference_steps, shift, device=device,
                                             dtype=torch.float32).tolist() + [0.0]
        target = torch.linspace(-0.4, 0.4, length * 64).reshape(1, length, 64)  # the fake model's x0
        for step, t in enumerate(sigmas[:-1]):
            if step == self.fail_at_step:
                raise RuntimeError("synthetic pipeline failure")
            xt = (xt.float() + (sigmas[step + 1] - t) * (latents.float() - target)).to(dtype)
            if callback_on_step_end is not None and self.call_callback:
                out = callback_on_step_end(self, step, t, {"latents": xt})
                if out is not None:
                    xt = out.pop("latents", xt)
            self.trace.append(xt.clone())
        if output_type != "latent":
            raise AssertionError("the server decodes itself; it must ask for latents")
        return SimpleNamespace(audios=xt)


class FakeHub:
    """huggingface_hub over a local cache holding the model's `snapshot` folder; None =
    not downloaded yet, where snapshot_download(local_files_only=True) raises."""

    def __init__(self, snapshot=None):
        self.snapshot, self.lookups = snapshot, 0

    def snapshot_download(self, repo_id, *, local_files_only):
        self.lookups += 1
        if not local_files_only:
            raise AssertionError("the revision lookup must never touch the network")
        if self.snapshot is None:
            raise FileNotFoundError("Cannot find an appropriate cached snapshot folder")
        return f"/hf/hub/models--{repo_id.replace('/', '--')}/snapshots/{self.snapshot}"


def wav_b64(samples, rate=48000, subtype="PCM_16"):
    buffer = io.BytesIO()
    sf.write(buffer, samples, rate, format="WAV", subtype=subtype)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def npy_b64(array):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def sf_pcm16(audio):
    """What libsndfile itself stores for float audio written as PCM_16."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, 48000, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return sf.read(buffer, dtype="int16", always_2d=True)[0]


def continuation_payload(context=40, generate=30, join=None, *, latents=True, gain=0.8, **spec):
    rng = np.random.default_rng(0)
    scaffold = np.zeros(((context + generate) * HOP, 2), dtype=np.int16)
    scaffold[:context * HOP] = rng.integers(-12000, 12000, (context * HOP, 2))
    body = {"protocol": "ace-latent-pin-v1", "context_frames": context, "generate_frames": generate,
            "join_frame": context - 13 if join is None else join, "crossfade_frames": 1, "latent_hop": HOP,
            "context_gain": None}
    payload = {"prompt": "fixture", "lyrics": "[instrumental]", "audio_duration": (context + generate) * HOP / 48000,
               "infer_step": 4, "guidance_scale": 7.0, "shift": 3.0, "seed": 7, "audio_b64": wav_b64(scaffold),
               "continuation": body}
    if latents:
        parent = (torch.randn(context, 64, generator=torch.Generator().manual_seed(5)) * 0.2).to(torch.bfloat16)
        body.update(context_gain=[[-1000, gain]], latents_model=MODEL, latents_model_revision=REVISION)
        payload["context_latents_b64"] = npy_b64(parent.float().numpy())
    body.update(spec)
    return payload, scaffold


def pin_inputs(context=30, generate=25):
    parent = torch.randn(1, context, 64, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16).float()
    noise = torch.randn(1, context + generate, 64, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    return parent, noise, dict(context=parent, noise=noise, context_frames=context, generate_frames=generate,
                               steps=6, shift=3.0, prompt="p", src_audio=torch.zeros(2, (context + generate) * HOP),
                               audio_duration=(context + generate) * HOP / 48000)


class PinTests(unittest.TestCase):
    def test_pin_follows_the_schedule_and_ends_exactly_on_the_parent(self):
        pipe = FakePipe()
        parent, noise, kwargs = pin_inputs()
        final = server._pinned_denoise(pipe, **kwargs)
        self.assertTrue(torch.equal(final[:, :30].float(), parent))
        schedule = pipe._get_timestep_schedule(6, 3.0, dtype=torch.float32).tolist()
        t_next = schedule[1:] + [0.0]
        for step, latents in enumerate(pipe.trace):
            expected = ((1 - t_next[step]) * parent + t_next[step] * noise[:, :30].float()).to(torch.bfloat16)
            self.assertTrue(torch.equal(latents[:, :30], expected), step)
        self.assertFalse(torch.equal(final[:, 30:], noise[:, 30:]))  # the window was denoised
        # Conditioning: the parent's latents before the window, the pipeline's silence inside it.
        self.assertTrue(torch.equal(pipe.conditioning[:, :30].float(), parent))
        self.assertTrue(torch.all(pipe.conditioning[:, 30:] == -0.5))
        call = pipe.calls[0]
        self.assertEqual((call["task_type"], call["output_type"], call["tensor_inputs"]), ("repaint", "latent", ["latents"]))
        self.assertIs(call["latents"], noise)
        self.assertEqual((int(call["repainting_start"] * 25), int(call["repainting_end"] * 25)), (30, 55))

    def test_schedule_is_the_pipelines_with_a_terminal_zero(self):
        schedule, t_next = server._pin_schedule(FakePipe(), 50, 3.0, torch.device("cpu"))
        self.assertEqual((len(schedule), len(t_next), schedule[0], t_next[-1]), (50, 50, 1.0, 0.0))
        self.assertEqual(t_next[:-1], schedule[1:])
        self.assertTrue(all(a > b for a, b in zip(t_next, t_next[1:], strict=False)))

    def test_prepare_src_latents_is_restored_even_when_the_call_raises(self):
        for fail_at in (None, 2):
            pipe = FakePipe(fail_at_step=fail_at)
            if fail_at is None:
                server._pinned_denoise(pipe, **pin_inputs()[2])
            else:
                with self.assertRaisesRegex(RuntimeError, "synthetic pipeline failure"):
                    server._pinned_denoise(pipe, **pin_inputs()[2])
            self.assertNotIn("prepare_src_latents", vars(pipe))
            self.assertIs(pipe.prepare_src_latents.__func__, FakePipe.prepare_src_latents)
        pipe = FakePipe(fail_at_step=1)
        own = pipe.prepare_src_latents = functools.partial(FakePipe.prepare_src_latents, pipe)
        with self.assertRaises(RuntimeError):
            server._pinned_denoise(pipe, **pin_inputs()[2])
        self.assertIs(pipe.prepare_src_latents, own)

    def test_a_pipeline_that_skips_the_callback_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, r"the pin ran after steps \[\]"):
            server._pinned_denoise(FakePipe(call_callback=False), **pin_inputs()[2])


class SignalTests(unittest.TestCase):
    def test_splice_keeps_the_scaffold_and_crossfades_linearly_over_one_frame(self):
        hop, rng = 8, np.random.default_rng(3)
        scaffold = rng.integers(-32768, 32767, (10 * hop, 2), dtype=np.int16, endpoint=True)
        rendered = rng.integers(-32768, 32767, (10 * hop, 2), dtype=np.int16, endpoint=True)
        out = server._splice_continuation(scaffold, rendered, 4, 1, hop)
        np.testing.assert_array_equal(out[:4 * hop + 1], scaffold[:4 * hop + 1])  # weight 0 at the join
        np.testing.assert_array_equal(out[5 * hop:], rendered[5 * hop:])
        for i in range(hop):
            a, b = scaffold[4 * hop + i].astype(float), rendered[4 * hop + i].astype(float)
            np.testing.assert_array_equal(out[4 * hop + i], np.rint(a + i / hop * (b - a)).astype(np.int16))
        ramp = server._splice_continuation(np.zeros((3 * hop, 1), np.int16), np.full((3 * hop, 1), 8000, np.int16), 1, 1, hop)
        self.assertEqual(ramp[:, 0].tolist(), [0] * hop + [1000 * i for i in range(hop)] + [8000] * hop)
        with self.assertRaises(ValueError):
            server._splice_continuation(scaffold, rendered.astype(np.int32), 4, 1, hop)

    def test_soft_clip_is_identity_below_the_knee_and_bounded_above(self):
        quiet = np.linspace(-0.95, 0.95, 20001, dtype=np.float32)
        np.testing.assert_array_equal(server._soft_clip(quiet), quiet)
        loud = np.array([0.9500001, 0.96, 1.0, 1.5, 4.0, 1e3], dtype=np.float32)
        clipped = server._soft_clip(np.concatenate([loud, -loud]))
        self.assertEqual(clipped.dtype, np.float32)
        self.assertTrue(np.all(np.abs(clipped) <= 1.0) and np.all(np.abs(clipped) > 0.95))
        np.testing.assert_array_equal(clipped[:6], -clipped[6:])
        self.assertTrue(np.all(np.diff(clipped[:6]) >= 0))
        self.assertAlmostEqual(float(clipped[0]), 0.9500001, places=6)  # slope 1 at the knee
        np.testing.assert_array_equal(server._soft_clip(loud), server._soft_clip(loud.copy()))

    def test_gain_envelope_interpolates_holds_after_the_cut_and_shifts(self):
        env = server._gain_envelope([(0, 1.0), (100, 2.0)], 300, 150)
        self.assertEqual(env.dtype, np.float32)
        self.assertAlmostEqual(float(env[50]), 1.5)
        self.assertTrue(np.all(env[100:] == np.float32(2.0)))
        ramp = server._gain_envelope([(-100, 1.0), (200, 4.0)], 300, 100)  # the cut inside a ramp
        np.testing.assert_allclose(ramp[:100], 1.0 + (np.arange(100) + 100) / 100.0, rtol=1e-6)
        self.assertTrue(np.all(ramp[100:] == np.float32(3.0)))
        np.testing.assert_array_equal(server._gain_envelope([(-480000, 0.7)], 10, 5), np.full(10, 0.7, np.float32))
        points = [(0, 1.0), (100, 2.0), (180, 0.5)]
        child = server._shift_envelope(points, 40, 150)
        self.assertEqual(child, [[0, 1.4], [60, 2.0], [110, 1.0625]])
        np.testing.assert_allclose(server._gain_envelope(child, 260, 110),
                                   server._gain_envelope(points, 300, 150)[40:], rtol=1e-6)
        self.assertEqual(server._shift_envelope([(-5000, 0.8)], 1000, 9000), [[0, 0.8]])
        self.assertEqual(server._shift_envelope([(0, 1.0), (500, 1.0)], 100, 400), [[0, 1.0]])

    def test_latents_npy_round_trip_from_bf16_is_exact(self):
        latents = torch.randn(37, 64).to(torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "job_000001.latents.npy"
            server._write_latents(path, latents)
            array = np.load(path, allow_pickle=False)
            self.assertEqual((array.dtype, array.shape), (np.float32, (37, 64)))
            self.assertTrue(torch.equal(torch.from_numpy(array), latents.float()))
            self.assertTrue(torch.equal(torch.from_numpy(array).to(torch.bfloat16), latents))
            back = server._read_context_latents(base64.b64encode(path.read_bytes()).decode(), 37)
            np.testing.assert_array_equal(back, array)
        with self.assertRaisesRegex(RuntimeError, "latents must be"):
            server._write_latents(Path("unused.npy"), torch.zeros(3, 63))


class ValidationTests(unittest.TestCase):
    def request(self, payload):
        return server._continuation_request(payload, model=MODEL, revision=REVISION)

    def assert_refused(self, pattern, payload):
        with self.assertRaisesRegex(ValueError, pattern):
            self.request(payload)

    def test_valid_requests_parse(self):
        request = self.request(continuation_payload()[0])
        self.assertEqual((request.context_frames, request.generate_frames, request.join_frame), (40, 30, 27))
        self.assertEqual((request.context_gain, request.latents.shape), ([(-1000, 0.8)], (40, 64)))
        audio = self.request(continuation_payload(latents=False)[0])
        self.assertEqual((audio.context_gain, audio.latents), ([(0, 1.0)], None))
        self.assertEqual(self.request(continuation_payload(join=29)[0]).join_frame, 29)  # 29 + 1 = 40 - 10
        mono, scaffold = continuation_payload()
        mono["audio_b64"] = wav_b64(scaffold[:, :1])
        np.testing.assert_array_equal(self.request(mono).scaffold, np.repeat(scaffold[:, :1], 2, axis=1))

    def test_bad_geometry_is_refused(self):
        self.assert_refused(r"must both be >= 25", continuation_payload(context=24, join=1)[0])
        self.assert_refused(r"must both be >= 25", continuation_payload(generate=24)[0])
        self.assert_refused(r"join_frame \(30\) \+ crossfade_frames \(1\) must be <= context_frames \(40\) - 10",
                            continuation_payload(join=30)[0])
        self.assert_refused(r"join_frame \(-1\)", continuation_payload(join=-1)[0])
        self.assert_refused(r"latent_hop 960", continuation_payload(latent_hop=960)[0])
        self.assert_refused(r"protocol 'v0'", continuation_payload(protocol="v0")[0])
        self.assert_refused(r"context_frames must be an integer", continuation_payload(context_frames=40.0)[0])
        payload = continuation_payload()[0]
        payload["audio_duration"] = 2.9
        self.assert_refused(r"audio_duration 2.9", payload)

    def test_bad_scaffolds_are_refused(self):
        payload, scaffold = continuation_payload()
        for audio, pattern in (
                (wav_b64(np.concatenate([scaffold, scaffold[:1]])), r"needs exactly \(context_frames \+ generate_frames\)"),
                (wav_b64(scaffold, rate=44100), r"48000 Hz PCM_16, got 44100 Hz"),
                (wav_b64(scaffold.astype(np.float32) / 32768, subtype="FLOAT"), r"PCM_16, got 48000 Hz FLOAT"),
                (wav_b64(np.repeat(scaffold[:, :1], 3, axis=1)), r"3 channels"),
                (wav_b64(np.ones_like(scaffold)), r"digital silence after the context"),
                (base64.b64encode(b"not a wav").decode(), r"not a readable WAV"),
                ("", r"requires audio_b64")):
            self.assert_refused(pattern, {**payload, "audio_b64": audio})

    def test_bad_latents_and_labels_are_refused(self):
        payload = continuation_payload()[0]
        bad = {"context_latents_b64": npy_b64(np.zeros((39, 64), np.float32))}
        self.assert_refused(r"float32 \[40, 64\], got float32 \[39, 64\]", {**payload, **bad})
        self.assert_refused(r"got float64", {**payload, "context_latents_b64": npy_b64(np.zeros((40, 64)))})
        self.assert_refused(r"non-finite", {**payload, "context_latents_b64": npy_b64(np.full((40, 64), np.nan, np.float32))})
        self.assert_refused(r"not a readable \.npy", {**payload, "context_latents_b64": base64.b64encode(b"x").decode()})
        self.assert_refused(r"come from other/model@0123456789abcdef; this renderer runs ACE-Step/test-model@0123",
                            continuation_payload(latents_model="other/model")[0])
        self.assert_refused(r"@deadbeef; this renderer", continuation_payload(latents_model_revision="deadbeef")[0])
        self.assert_refused(r"need continuation.context_gain", continuation_payload(context_gain=None)[0])
        self.assert_refused(r"need continuation.context_gain", continuation_payload(latents_model="")[0])
        self.assert_refused(r"samples must increase", continuation_payload(context_gain=[[0, 1.0], [0, 2.0]])[0])
        self.assert_refused(r"positive finite", continuation_payload(context_gain=[[0, 0.0]])[0])
        self.assert_refused(r"sample must be an integer", continuation_payload(context_gain=[[0.5, 1.0]])[0])
        self.assert_refused(r"belong to a request that sends context_latents_b64",
                            continuation_payload(latents=False, context_gain=[[0, 1.0]])[0])

    def test_segments_and_continuation_together_are_refused(self):
        self.assertEqual([server._operation(p) for p in ({}, {"segments": []}, {"segments": [[0, 1]]}, {"continuation": {}})],
                         ["generate", "generate", "repaint", "continuation"])
        with self.assertRaisesRegex(ValueError, "never both"):
            server._operation({"continuation": {}, "segments": [[0, 1]]})


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.states = []

    def tearDown(self):
        for state in self.states:
            state.receipts.close()
        self.directory.cleanup()

    def runtime(self, pipe, revision=REVISION):
        runtime = server.AceRuntime.__new__(server.AceRuntime)
        runtime.output_dir, runtime.engine, runtime.dtype, runtime.checkpoint_dir = self.root, "v15", "bfloat16", ""
        runtime._lock, runtime._model_id, runtime._model_revision_cache = threading.Lock(), MODEL, revision
        runtime._pipe, runtime._embed_model_id = pipe, "synthetic-embedder"
        runtime._load = lambda: pipe
        return runtime

    def state(self, pipe, revision=REVISION):
        with patch.object(threading.Thread, "start"), patch.dict(os.environ, {"ACE_WATCHDOG": "0"}):
            state = server.AppState(self.runtime(pipe, revision))
        self.states.append(state)
        return state

    def dispatch(self, state, payload):
        """One job through the real enqueue -> AppState._loop dispatch -> receipt."""
        job = state.enqueue({"operation_id": "7f1c1c56-5a8e-4a43-9a43-07c1e8d5a0f1", **payload})
        with patch.object(state._queue, "task_done", side_effect=StopIteration), self.assertRaises(StopIteration):
            state._loop()
        return state.get(job.id)

    def run_job(self, payload, pipe, revision=REVISION):
        state = self.state(pipe, revision)
        return state, self.dispatch(state, payload)

    def unloaded(self, pipe, hub, downloads):
        """A server before its real _load(), and the modules it imports: from_pretrained
        leaves snapshot `downloads` in the hub's cache (None = still none) and returns pipe."""
        def from_pretrained(model_id, dtype):
            self.assertEqual((model_id, dtype), (MODEL, torch.bfloat16))
            hub.snapshot = downloads
            return SimpleNamespace(to=lambda device: pipe)

        state = self.state(None, revision="")
        del state.runtime._load  # the real one
        diffusers = SimpleNamespace(AceStepPipeline=SimpleNamespace(from_pretrained=from_pretrained))
        return state, patch.dict(sys.modules, {"huggingface_hub": hub, "diffusers": diffusers})

    def test_continuation_from_stored_latents(self):
        pipe = FakePipe()
        payload, scaffold = continuation_payload(gain=3.0)
        state, job = self.run_job(payload, pipe)
        self.assertEqual(job.status, "done", job.error)
        final = pipe.trace[-1]
        parent = np.load(io.BytesIO(base64.b64decode(payload["context_latents_b64"])))
        np.testing.assert_array_equal(final[0, :40].float().numpy(), parent)
        unclipped = final[0, :, :2].float().numpy().repeat(HOP, axis=0) * np.float32(3.0)
        rendered = server._soft_clip(unclipped)
        expected = sf_pcm16(rendered)
        out, rate = sf.read(job.audios[0].path, dtype="int16", always_2d=True)
        join, xf = 27 * HOP, 28 * HOP
        self.assertEqual((rate, out.shape), (48000, (70 * HOP, 2)))
        np.testing.assert_array_equal(out[:join + 1], scaffold[:join + 1])
        np.testing.assert_array_equal(out[xf:], expected[xf:])
        weight = (np.arange(HOP) / HOP)[:, None]
        a = scaffold[join:xf].astype(float)
        np.testing.assert_array_equal(out[join:xf], np.rint(a + weight * (expected[join:xf] - a)).astype(np.int16))
        stored = np.load(job.latents_path)
        np.testing.assert_array_equal(stored, final[0, 27:].float().numpy())
        self.assertEqual(job.latents, {
            "format": "ace-latents-v1", "url": f"/jobs/{job.id}/latents", "frames": 43, "latent_hop": 1920,
            "sample_rate": 48000, "channels": 64, "gain_envelope": [[0, 3.0]], "soft_clip_knee": 0.95,
            "model": MODEL, "model_revision": REVISION, "render_start_frame": 27})
        diagnostics = job.continuation
        self.assertEqual({k: diagnostics[k] for k in ("context_source", "join_sample", "crossfade_samples")},
                         {"context_source": "latents", "join_sample": join, "crossfade_samples": HOP})
        self.assertAlmostEqual(diagnostics["join_max_abs_diff"], float(np.abs(a / 32768 - rendered[join:xf]).max()))
        self.assertAlmostEqual(diagnostics["new_peak_before_clip"], float(np.abs(unclipped[40 * HOP:]).max()), places=6)
        self.assertEqual(diagnostics["soft_clipped_samples"], int((np.abs(unclipped[join:]) > np.float32(0.95)).sum()))
        self.assertGreater(diagnostics["soft_clipped_samples"], 0)
        body = job.payload_dict()
        self.assertEqual((body["latents"], body["continuation"]["context_source"]), (job.latents, "latents"))
        self.assertEqual(set(job.output_identity), {"wav", "waveform", "latents"})
        self.assertEqual(job.output_identity["latents"]["byte_count"], Path(job.latents_path).stat().st_size)
        restored = server.Job.restore(state.receipts.all()[0])  # what a restarted server serves
        self.assertEqual((restored.latents, restored.latents_path, restored.continuation),
                         (job.latents, job.latents_path, job.continuation))
        status, data, kind = self.get(state, f"/jobs/{job.id}/latents")
        self.assertEqual((status, kind, data), (200, "application/octet-stream", Path(job.latents_path).read_bytes()))
        Path(job.latents_path).unlink()
        self.assertEqual(job.payload_dict()["status"], "output_expired")
        self.assertEqual(self.get(state, f"/jobs/{job.id}/latents")[0], 404)

    def test_continuation_from_audio_encodes_the_posterior_mean(self):
        pipe = FakePipe()
        payload, scaffold = continuation_payload(latents=False)
        _state, job = self.run_job(payload, pipe)
        self.assertEqual(job.status, "done", job.error)
        context = torch.from_numpy(scaffold[:40 * HOP].T.astype(np.float32) / 32768).unsqueeze(0)
        mean = pipe.vae.encode(context.to(torch.bfloat16)).latent_dist.mode().transpose(1, 2)
        self.assertTrue(torch.equal(pipe.trace[-1][:, :40], mean))
        self.assertEqual(job.continuation["context_source"], "audio")
        self.assertEqual(job.latents["gain_envelope"], [[0, 1.0]])
        out = sf.read(job.audios[0].path, dtype="int16", always_2d=True)[0]
        np.testing.assert_array_equal(out[:27 * HOP], scaffold[:27 * HOP])

    def test_failed_validation_fails_the_job_and_writes_nothing(self):
        _state, job = self.run_job(continuation_payload(latents_model="other/model")[0], FakePipe())
        self.assertEqual(job.status, "error")
        self.assertIn("context latents come from other/model", job.error)
        self.assertEqual((job.audios, job.latents_path, list(self.root.glob("job_*"))), ([], "", []))

    def test_text2music_stores_the_pipeline_level_and_its_latents(self):
        pipe = FakePipe()
        _state, job = self.run_job({"prompt": "fixture", "audio_duration": 2.0, "infer_step": 3}, pipe)
        self.assertEqual(job.status, "done", job.error)
        final = pipe.trace[-1]
        decoded = final[0, :, :2].float().numpy().repeat(HOP, axis=0)
        gain = server.TARGET_PEAK / float(np.abs(decoded).max())
        out = sf.read(job.audios[0].path, dtype="int16", always_2d=True)[0]
        self.assertEqual(out.shape, (50 * HOP, 2))
        np.testing.assert_array_equal(out, sf_pcm16(decoded * np.float32(gain)))  # the clip is identity here
        np.testing.assert_array_equal(np.load(job.latents_path), final[0].float().numpy())
        self.assertEqual((job.latents["frames"], job.latents["render_start_frame"]), (50, 0))
        self.assertEqual(job.latents["gain_envelope"], [[0, gain]])
        self.assertEqual(job.continuation, {})
        self.assertEqual(pipe.calls[0]["output_type"], "latent")

    def get(self, state, path):
        handler = server.Handler.__new__(server.Handler)
        handler.path = path
        result = []
        handler._json = lambda data, status=200: result.append((status, data, "application/json"))
        handler._bytes = lambda data, kind, status=200: result.append((status, data, kind))
        with patch.object(server, "STATE", state), patch.object(server, "gpu_status", return_value={}):
            handler.do_GET()
        return result[0]

    def test_health_advertises_capabilities_only_for_v15(self):
        with patch.object(threading.Thread, "start"), patch.dict(os.environ, {"ACE_WATCHDOG": "0"}):
            state = server.AppState(self.runtime(FakePipe()))
        self.states.append(state)
        body = self.get(state, "/health")[1]
        self.assertEqual(body["capabilities"], ["continuation:ace-latent-pin-v1", "latents:ace-latents-v1"])
        self.assertEqual((body["operation_protocol"], body["model_revision"]), ("durable-v1", REVISION))
        state.runtime.engine = "v1"
        self.assertEqual(self.get(state, "/health")[1]["capabilities"], [])

    def test_enqueue_refuses_segments_with_continuation(self):
        with patch.object(threading.Thread, "start"), patch.dict(os.environ, {"ACE_WATCHDOG": "0"}):
            state = server.AppState(self.runtime(FakePipe()))
        self.states.append(state)
        with self.assertRaisesRegex(ValueError, "never both"):
            state.enqueue({**continuation_payload()[0], "segments": [[0, 1]]})
        self.assertTrue(state._queue.empty())

    def test_pruner_keeps_unacknowledged_latents(self):
        pipe = FakePipe()
        state, job = self.run_job({"prompt": "fixture", "audio_duration": 2.0, "infer_step": 2}, pipe)
        latents = Path(job.latents_path)
        stray = self.root / "embed_1.wav"
        stray.write_bytes(b"temp")
        old = time.time() - 3600
        for path in (latents, stray):
            os.utime(path, (old, old))
        runtime = server.AceRuntime.__new__(server.AceRuntime)
        runtime.output_dir, runtime.receipts = self.root, state.receipts
        callbacks = []
        thread = lambda **kwargs: SimpleNamespace(start=lambda: callbacks.append(kwargs["target"]))  # noqa: E731
        with patch.object(threading, "Thread", side_effect=thread):
            runtime._start_output_pruner()
        with patch.object(server.time, "sleep", side_effect=StopIteration), self.assertRaises(StopIteration):
            callbacks[0]()
        self.assertTrue(latents.exists())
        self.assertFalse(stray.exists())
        state.receipts.acknowledge(job.operation_id, job.request_sha256)
        with patch.object(server.time, "sleep", side_effect=StopIteration), self.assertRaises(StopIteration):
            callbacks[0]()
        self.assertFalse(latents.exists())

    def test_a_failed_revision_lookup_is_retried_not_cached(self):
        """A lookup that missed (e.g. /health polled during the preload, before the snapshot
        exists) used to cache "" and fail every later text2music and continuation until a
        restart."""
        hub = FakeHub()
        state = self.state(FakePipe(), revision="")
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            self.assertEqual(self.get(state, "/health")[1]["model_revision"], "")
            self.assertEqual(self.get(state, "/health")[1]["model_revision"], "")
            hub.snapshot = SNAPSHOT  # the preload's download finished
            job = self.dispatch(state, {"prompt": "fixture", "audio_duration": 2.0, "infer_step": 2})
            self.assertEqual(self.get(state, "/health")[1]["model_revision"], REVISION)
        self.assertEqual(job.status, "done", job.error)
        self.assertEqual(job.latents["model_revision"], REVISION)
        self.assertEqual(hub.lookups, 3)  # two misses, one hit, then the cache

    def test_the_load_resolves_the_revision_it_loaded(self):
        """A lookup before the load can see an older snapshot than the one the load downloads:
        the load looks again, and a continuation is checked against what was loaded."""
        old, new = "1" * 40, "2" * 40
        pipe, hub = FakePipe(), FakeHub(old)
        state, modules = self.unloaded(pipe, hub, downloads=new)
        with modules:
            self.assertEqual(self.get(state, "/health")[1]["model_revision"], old[:16])
            self.assertEqual(state.runtime._model_revision_cache, "")  # a lookup before the load is not kept
            job = self.dispatch(state, continuation_payload(latents_model_revision=old[:16])[0])
            self.assertEqual(self.get(state, "/health")[1]["model_revision"], new[:16])
        self.assertIs(state.runtime._pipe, pipe)
        self.assertEqual(job.status, "error")
        self.assertIn(f"come from {MODEL}@{old[:16]}; this renderer runs {MODEL}@{new[:16]}", job.error)
        self.assertEqual((pipe.calls, list(self.root.glob("job_*"))), ([], []))

    def test_a_revision_the_load_cannot_find_fails_the_preload(self):
        """The preload reports it (the app routes no work to a box with a preload_error) instead
        of every later job; the pipeline stays loaded for repaint, which needs no revision."""
        pipe = FakePipe()
        state, modules = self.unloaded(pipe, FakeHub(), downloads=None)
        with modules:
            state._preload_run()
            health = self.get(state, "/health")[1]
        self.assertTrue(health["preload_error"].startswith(
            f"RuntimeError: no HF snapshot revision found for {MODEL} (FileNotFoundError: "), health["preload_error"])
        self.assertEqual((health["loaded"], health["preloading"], health["model_revision"]), (True, False, ""))
        self.assertIs(state.runtime._pipe, pipe)

    def assert_fails_before_rendering(self, payload):
        pipe = FakePipe()
        with patch.dict(sys.modules, {"huggingface_hub": FakeHub()}):
            _state, job = self.run_job(payload, pipe, revision="")
        self.assertEqual((job.status, len(pipe.calls), list(self.root.glob("job_*"))), ("error", 0, []))
        self.assertEqual((job.audios, job.latents_path), ([], ""))
        self.assertIn(f"no HF snapshot revision found for {MODEL} (FileNotFoundError", job.error)

    def test_text2music_without_a_revision_renders_and_writes_nothing(self):
        """The revision used to be looked up after the GPU render and the WAV, which the pruner
        then kept forever (an error receipt is never acknowledged)."""
        self.assert_fails_before_rendering({"prompt": "fixture", "audio_duration": 2.0, "infer_step": 2})

    def test_continuation_without_a_revision_renders_and_writes_nothing(self):
        self.assert_fails_before_rendering(continuation_payload()[0])


if __name__ == "__main__":
    unittest.main()
