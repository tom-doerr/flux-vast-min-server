"""In-process RIFE 8x frame interpolation (Practical-RIFE v4.25, MIT license).

Vendored arch + weights live under ``rife/`` (rife/train_log/{RIFE_HDv3,
IFNet_HDv3,refine}.py + flownet.pkl, rife/model/warplayer.py). The flownet is
loaded ONCE (module-global, lock-guarded) and reused for every clip -- no
per-clip subprocess / CUDA-init overhead.

Wan generates 16fps clips; 8x arbitrary-timestep interpolation -> 128fps.
RIFE 4.x supports arbitrary-t inference directly, so 8x = 7 synthesized frames
(t = 1/8 .. 7/8) between every consecutive pair, plus the originals. Frame
count: (N-1)*8 + 1.

No silent fallback: a clip that cannot be read/interpolated raises, so the
caller surfaces the failure instead of shipping an un-interpolated clip as if
it were interpolated.
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v3 as iio

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOCK = threading.Lock()
_MODEL: Any = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model() -> Any:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from rife.train_log.RIFE_HDv3 import Model  # noqa: E402

    model = Model()
    model.load_model(os.path.join(_HERE, "rife", "train_log"), -1)
    model.eval()
    model.device()
    _MODEL = model
    return _MODEL


def interpolate_file(
    in_path: str,
    out_path: str,
    *,
    multi: int = 8,
    out_fps: int = 128,
    scale: float = 1.0,
    crf: int = 18,
) -> dict[str, Any]:
    """Read in_path, RIFE-interpolate multi-x, write out_path at out_fps.
    Returns render stats. Raises on any failure (never writes a partial or
    un-interpolated clip as if it succeeded)."""
    if multi < 2:
        raise ValueError(f"RIFE multi must be >= 2, got {multi}")
    frames = list(iio.imiter(in_path, plugin="pyav"))
    if len(frames) < 2:
        raise RuntimeError(f"RIFE: too few frames ({len(frames)}) in {in_path}")

    h, w = frames[0].shape[:2]
    tmp = max(128, int(128 / scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    pad = (0, pw - w, 0, ph - h)

    def to_tensor(frame: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(frame)).to(_DEVICE)
        t = t.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return F.pad(t, pad)

    def to_frame(t: torch.Tensor) -> np.ndarray:
        t = t[:, :, :h, :w].clamp(0.0, 1.0)[0]
        return (t.permute(1, 2, 0) * 255.0).byte().cpu().numpy()

    model = _load_model()
    out_frames: list[np.ndarray] = []
    with _LOCK, torch.no_grad():
        i0 = to_tensor(frames[0])
        out_frames.append(np.ascontiguousarray(frames[0]))
        for k in range(1, len(frames)):
            i1 = to_tensor(frames[k])
            for i in range(multi - 1):
                mid = model.inference(i0, i1, (i + 1) * 1.0 / multi, scale)
                out_frames.append(to_frame(mid))
            out_frames.append(np.ascontiguousarray(frames[k]))
            i0 = i1

    arr = np.stack(out_frames)
    iio.imwrite(
        out_path,
        arr,
        plugin="pyav",
        fps=out_fps,
        codec="libx264",
        out_pixel_format="yuv420p",
    )
    if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
        raise RuntimeError(f"RIFE: output not written to {out_path}")
    return {
        "frames_in": len(frames),
        "frames_out": len(out_frames),
        "fps": out_fps,
        "multi": multi,
        "width": w,
        "height": h,
    }


if __name__ == "__main__":
    import json
    import time

    src, dst = sys.argv[1], sys.argv[2]
    mult = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    fps = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    t0 = time.time()
    stats = interpolate_file(src, dst, multi=mult, out_fps=fps)
    stats["seconds"] = round(time.time() - t0, 2)
    print(json.dumps(stats))
