"""Convolution kernels and gradient preconditioning.

Ports pjsr/lib/fftalign.jsh:14-33 (KROON + gradientMagnitude) and the
Laplacian sharpness kernel of pjsr/ser-stack.js:209-218.

cv2.filter2D computes correlation, not convolution. KROON is vertically
antisymmetric and the Laplacian is symmetric, and both uses square or take
a scale-free statistic of the response, so the flip does not affect any
result here.
"""

import cv2
import numpy as np

# Kroon 5x5 optimized derivative filter (y-derivative orientation).
KROON = np.array(
    [
        [+0.0007, +0.0052, +0.0370, +0.0052, +0.0007],
        [+0.0037, +0.1187, +0.2589, +0.1187, +0.0037],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.0037, -0.1187, -0.2589, -0.1187, -0.0037],
        [-0.0007, -0.0052, -0.0370, -0.0052, -0.0007],
    ],
    dtype=np.float32,
)

LAPLACIAN = np.array(
    [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32
)


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """sqrt(Ix^2 + Iy^2) with the Kroon derivative pair.

    Reduces phase-correlation sensitivity to illumination differences
    (the reason fftalign.jsh preconditions with it).
    """
    src = np.asarray(img, dtype=np.float32)
    gy = cv2.filter2D(src, cv2.CV_32F, KROON)
    gx = cv2.filter2D(src, cv2.CV_32F, KROON.T)
    return cv2.sqrt(gx * gx + gy * gy)


def laplacian_sharpness(img: np.ndarray) -> float:
    """Std dev of the 3x3 Laplacian response (frame quality metric).

    Translation-invariant, monotonic with seeing quality at fixed target
    (ser-stack.js:213-218). Border handling differs from PI — fine for a
    ranking metric.
    """
    src = np.asarray(img, dtype=np.float32)
    return float(cv2.filter2D(src, cv2.CV_32F, LAPLACIAN).std())
