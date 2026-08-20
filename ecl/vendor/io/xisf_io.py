"""XISF read/write.

XISF stays the canonical intermediate format for coexistence with the
PixInsight pipeline: PI's lunar-finish.js must read Python-written
_stack.xisf, and the lunation assembler must read the existing
out/finished/FIN_*.xisf archive.

Read/write via the `xisf` PyPI package (monolithic XISF). Images are
normalized to float32, mono as (H, W), RGB as (H, W, 3).
Write is verified against PixInsight itself (M0 gate); if that ever
regresses, replace with a first-party monolithic writer (~150 lines).
"""

import numpy as np
from xisf import XISF


def read_xisf(path: str) -> np.ndarray:
    """First image of a monolithic XISF as float32, (H,W) or (H,W,3)."""
    f = XISF(path)
    im = f.read_image(0)
    im = np.asarray(im)
    if im.ndim == 3 and im.shape[2] == 1:
        im = im[:, :, 0]
    if im.dtype == np.uint16:
        return im.astype(np.float32) / 65535.0
    if im.dtype == np.uint8:
        return im.astype(np.float32) / 255.0
    return im.astype(np.float32)


def write_xisf(path: str, image: np.ndarray) -> None:
    """Write float32 mono/RGB as a monolithic XISF PixInsight can open."""
    im = np.asarray(image, dtype=np.float32)
    if im.ndim == 2:
        im = im[:, :, np.newaxis]
    XISF.write(path, im, creator_app="lunation")
