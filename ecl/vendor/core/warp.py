"""Sub-pixel translate / rotate / resample.

Conventions (locked by tests/test_warp.py — do not change one without the
other):
- translate(img, dx, dy): image CONTENT moves by +dx (right) and +dy (down).
- rotate(img, angle, cx, cy): positive angle rotates content visually
  counter-clockwise about the explicit pivot — PI's Image.rotate convention
  (CLAUDE.md: positive = visual CCW). "Visually CCW" is measured with the
  math convention on screen axes: theta = atan2(cy - y, x - cx).
- resample(img, scale): output dimensions round(w*scale) x round(h*scale).

cv2 is the primary engine; method="ndimage" is the A/B alternative for
drizzle-placement quality comparisons (plan risk #2).
"""

from typing import Literal

import cv2
import numpy as np
from scipy import ndimage

Interp = Literal["nearest", "bilinear", "bicubic", "mitchell", "lanczos"]

_CV2_INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    # PI BicubicSpline != cv2 INTER_CUBIC (Keys a=-0.75) and cv2 has no
    # Mitchell — accepted under the better-than bar; A/B via method="ndimage".
    "bicubic": cv2.INTER_CUBIC,
    "mitchell": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}

_NDIMAGE_ORDER = {"nearest": 0, "bilinear": 1, "bicubic": 3, "mitchell": 3,
                  "lanczos": 3}


def translate(
    img: np.ndarray,
    dx: float,
    dy: float,
    interp: Interp = "bicubic",
    method: str = "cv2",
) -> np.ndarray:
    src = np.asarray(img, dtype=np.float32)
    if method == "ndimage":
        return ndimage.shift(
            src, (dy, dx), order=_NDIMAGE_ORDER[interp], mode="constant", cval=0.0
        ).astype(np.float32)
    m = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float64)
    return cv2.warpAffine(
        src, m, (src.shape[1], src.shape[0]),
        flags=_CV2_INTERP[interp], borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )


def rotate(
    img: np.ndarray,
    angle: float,
    cx: float,
    cy: float,
    interp: Interp = "bicubic",
) -> np.ndarray:
    """Rotate content by `angle` radians, positive = visual CCW, about (cx, cy)."""
    src = np.asarray(img, dtype=np.float32)
    m = cv2.getRotationMatrix2D((cx, cy), np.degrees(angle), 1.0)
    return cv2.warpAffine(
        src, m, (src.shape[1], src.shape[0]),
        flags=_CV2_INTERP[interp], borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )


def resample(
    img: np.ndarray, scale: float, interp: Interp = "bicubic"
) -> np.ndarray:
    src = np.asarray(img, dtype=np.float32)
    w = int(round(src.shape[1] * scale))
    h = int(round(src.shape[0] * scale))
    flag = _CV2_INTERP[interp]
    # PI's Mitchell is an anti-aliasing cubic; cv2 INTER_CUBIC aliases on
    # downscale, INTER_AREA is the equivalent-quality choice there
    if interp == "mitchell" and scale < 1:
        flag = cv2.INTER_AREA
    return cv2.resize(src, (w, h), interpolation=flag)
