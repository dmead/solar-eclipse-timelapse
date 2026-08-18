"""Image I/O — the layer that used to be PixInsight's `ImageWindow`.

Every PJSR script in this pipeline read images the same way:

    ImageWindow.open(path) -> mainView.image.getSamples(Float32Array, Rect, ch)

and wrote them back through `new ImageWindow` / `setSamples` / `saveAs`. That is
the entire reason PixInsight was on the critical path for the corona chain — the
processing in between was always plain arithmetic on Float32Array planes.

XISF survives the port. lunation reads and writes it with the pure-Python `xisf`
package (verified bit-exact against PixInsight's own reader), so the corona
stage boundaries do not move and nothing needs converting. Nothing outside PJSR
ever opened one anyway.

Convention here, matching lunation: mono images are (H, W), colour images are
(H, W, 3), float32, nominally [0, 1] but linear stacks routinely exceed 1 and are
never clipped on the way through.
"""

import os

import numpy as np

from lunation.io.images import read_image, write_png, write_tiff32
from lunation.io.xisf_io import read_xisf, write_xisf

__all__ = ["read", "write", "read_xisf", "write_xisf", "read_image",
           "write_png", "write_tiff32", "channels", "stack_channels",
           "to_display"]


def _ensure_dir(path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def read(path):
    """Read any format this pipeline produces, dispatched on extension."""
    if str(path).lower().endswith(".xisf"):
        return read_xisf(str(path))
    return read_image(str(path))


def write(path, image, bit_depth=16):
    """Write dispatched on extension. `bit_depth` applies to PNG only.

    XISF and TIFF stay float32 — they are intermediates and must not be
    quantised. PNG is a deliverable or a preview and gets 8 or 16 bits.
    """
    path = str(path)
    _ensure_dir(path)
    lower = path.lower()
    a = np.asarray(image, dtype=np.float32)
    if lower.endswith(".xisf"):
        write_xisf(path, a)
    elif lower.endswith((".tif", ".tiff")):
        write_tiff32(path, a)
    elif lower.endswith(".png"):
        write_png(path, a, bit_depth=bit_depth)
    else:
        raise ValueError(f"unsupported output format: {path}")
    return path


def channels(image):
    """Split (H, W, 3) into a list of (H, W) planes; pass mono through as [img]."""
    a = np.asarray(image)
    if a.ndim == 2:
        return [a]
    if a.ndim == 3 and a.shape[2] == 3:
        return [a[:, :, c] for c in range(3)]
    raise ValueError(f"expected (H,W) or (H,W,3), got shape {a.shape}")


def stack_channels(r, g, b):
    """Three (H, W) planes -> one (H, W, 3) colour image."""
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def to_display(image, gamma=0.65, black=None, white=None):
    """Linear -> display-referred, for previews only.

    Deliberately separate from anything that writes an intermediate: the corona
    spans the chromosphere near 1.0 and the outer corona near 0.001, so a
    rescale-and-gamma preview crushes everything interesting. It is a look at
    the data, never an input to a later stage.
    """
    a = np.asarray(image, dtype=np.float32)
    lo = np.min(a) if black is None else black
    hi = np.max(a) if white is None else white
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0) ** np.float32(gamma)
