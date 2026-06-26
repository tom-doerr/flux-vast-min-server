"""Tests for the adversarial-projection upscaler hosted on the gen server.

Covers the pure-numpy L2 projection and the upscale() loop driven by a stubbed
pipeline (no GPU / no model load), so CI can verify the contract and that each
PGD step stays within the L2 ball.
"""
import base64
import io
import threading

import numpy as np
from PIL import Image

import flux_vast_min_server as S


def test_rms_zero_for_identical():
    a = np.zeros((4, 4, 3))
    assert S.rms(a) == 0.0


def test_project_l2_zero_radius_returns_orig():
    orig = np.full((3, 3, 3), 0.5)
    x = orig + 0.3
    out = S.project_l2(x, orig, 0.0)
    assert np.allclose(out, orig)


def test_project_l2_below_radius_is_noop():
    orig = np.zeros((8, 8, 3))
    x = np.full((8, 8, 3), 0.01)
    out = S.project_l2(x, orig, 0.5)
    assert np.allclose(out, x)


def test_project_l2_above_radius_scales_to_radius():
    orig = np.zeros((8, 8, 3))
    x = np.full((8, 8, 3), 1.0)
    out = S.project_l2(x, orig, 0.05)
    assert abs(S.rms(out - orig) - 0.05) < 1e-9


def test_project_l2_resolution_independent():
    small = S.project_l2(np.ones((4, 4, 3)), np.zeros((4, 4, 3)), 0.1)
    big = S.project_l2(np.ones((64, 64, 3)), np.zeros((64, 64, 3)), 0.1)
    assert abs(S.rms(small) - S.rms(big)) < 1e-9


def _runtime_with_fake_pipe():
    rt = object.__new__(S.FluxRuntime)
    rt._generate_lock = threading.RLock()

    def pipe_factory(model_id):
        def _call(image=None, prompt="", num_inference_steps=6,
                  guidance_scale=3.5, output_type="pil"):
            arr = np.asarray(image[0]).astype(np.float64) + 40
            arr = np.clip(arr, 0, 255).astype("uint8")
            return type("O", (), {"images": [Image.fromarray(arr, "RGB")]})()
        return _call

    rt.pipe = pipe_factory
    return rt


def _b64_image(w=32, h=32):
    rng = np.random.RandomState(0).rand(h, w, 3)
    im = Image.fromarray((rng * 255).astype("uint8"), "RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_upscale_scales_and_bounds_l2():
    rt = _runtime_with_fake_pipe()
    res = rt.upscale({"image_b64": _b64_image(32, 32), "prompt": "",
                      "num_steps": 3, "l2_radius": 0.05, "gain": 2.0,
                      "scale_factor": 2})
    assert res["width"] == 64 and res["height"] == 64
    assert res["final_l2"] <= 0.05 + 1e-6
    assert set(res) >= {"image_b64", "width", "height", "took_ms",
                        "final_l2", "num_steps", "l2_radius"}
    out = Image.open(io.BytesIO(base64.b64decode(res["image_b64"])))
    assert out.size == (64, 64)


def test_upscale_requires_image_b64():
    import pytest
    rt = _runtime_with_fake_pipe()
    with pytest.raises(RuntimeError, match="image_b64 is required"):
        rt.upscale({})
