"""Conventional image formats: TIFF/PNG/JPG read, PNG/TIFF write.

Replaces PJSR ImageWindow.open for the prep-finished ingest path. All
reads return float32 [0,1], mono (H,W) or RGB (H,W,3) in RGB channel
order (cv2's BGR is swapped on the way in/out).
"""

import os

import cv2
import numpy as np
import tifffile


def _normalize(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    if a.dtype == np.uint16:
        return a.astype(np.float32) / 65535.0
    return a.astype(np.float32)


def read_image(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        try:
            a = tifffile.imread(path)
        except ValueError:
            # compressed variants (LZW etc.) without imagecodecs -> cv2
            a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if a is None:
                raise
            if a.ndim == 3:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        if a.ndim == 3 and a.shape[2] > 3:
            a = a[:, :, :3]
        return _normalize(a)
    if ext == ".xisf":
        from .xisf_io import read_xisf

        return read_xisf(path)
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise IOError(f"cannot read image: {path}")
    if a.ndim == 3:
        code = cv2.COLOR_BGRA2RGB if a.shape[2] == 4 else cv2.COLOR_BGR2RGB
        a = cv2.cvtColor(a, code)
    return _normalize(a)


def write_png(path: str, image: np.ndarray, bit_depth: int = 8) -> None:
    im = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if bit_depth == 16:
        q = np.rint(im * 65535.0).astype(np.uint16)
    else:
        q = np.rint(im * 255.0).astype(np.uint8)
    if q.ndim == 3:
        q = cv2.cvtColor(q, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(path, q):
        raise IOError(f"cannot write image: {path}")


def write_tiff32(path: str, image: np.ndarray) -> None:
    """Debug/side-channel float TIFF (never the pipeline contract)."""
    tifffile.imwrite(path, np.asarray(image, dtype=np.float32))
