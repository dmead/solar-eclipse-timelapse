"""FFT phase-correlation translation engine.

Port of pjsr/lib/fftalign.jsh (after PixInsight's FFTRegistration.js,
translation path only), with the correlation/peak math delegated to
skimage.registration.phase_cross_correlation (Guizar-Sicairos upsampled
DFT) by default. engine="ported" is the 1:1 fftalign port kept for A/B
fallback (plan risk #3b).

Convention (locked by tests/test_fftreg.py):
    evaluate(target) returns (dx, dy) = displacement of target relative to
    the reference; translate(target, -dx, -dy) aligns it to the reference.
"""

import numpy as np
import scipy.fft as sfft
from skimage.registration import phase_cross_correlation

from .kernels import gradient_magnitude


class PhaseCorrelator:
    """initialize(ref) once, then evaluate(target) per frame."""

    def __init__(
        self,
        use_gradient: bool = False,
        engine: str = "skimage",
        upsample: int = 100,
    ):
        self.use_gradient = use_gradient
        self.engine = engine
        self.upsample = upsample
        self._ref = None       # preconditioned reference (skimage engine)
        self._f0 = None        # reference spectrum (ported engine)
        self._size = 0

    def _precondition(self, img: np.ndarray) -> np.ndarray:
        src = np.asarray(img, dtype=np.float32)
        return gradient_magnitude(src) if self.use_gradient else src

    def initialize(self, ref: np.ndarray) -> None:
        pre = self._precondition(ref)
        if self.engine == "skimage":
            self._ref = pre
        else:
            self._size = sfft.next_fast_len(max(pre.shape))
            self._f0 = sfft.fft2(self._padded(pre), workers=-1)

    def evaluate(self, target: np.ndarray) -> tuple[float, float]:
        pre = self._precondition(target)
        if self.engine == "skimage":
            # skimage returns the (row, col) shift required to register the
            # moving image onto the reference, i.e. minus the displacement.
            shift, _err, _dc = phase_cross_correlation(
                self._ref, pre, upsample_factor=self.upsample, normalization="phase"
            )
            return -float(shift[1]), -float(shift[0])
        return self._evaluate_ported(pre)

    # --- ported engine: 1:1 fftalign.jsh:49-104 ---

    def _padded(self, img: np.ndarray) -> np.ndarray:
        n = self._size
        out = np.zeros((n, n), dtype=np.float32)
        y0 = (n - img.shape[0]) >> 1
        x0 = (n - img.shape[1]) >> 1
        out[y0 : y0 + img.shape[0], x0 : x0 + img.shape[1]] = img
        return out

    def _evaluate_ported(self, pre: np.ndarray) -> tuple[float, float]:
        f1 = sfft.fft2(self._padded(pre), workers=-1)
        # target = ref shifted by +s  =>  ifft2(F1 * conj(F0)) peaks at s
        cps = f1 * np.conj(self._f0)
        mag = np.abs(cps)
        cps /= np.where(mag > 0, mag, 1.0)
        r = np.real(sfft.ifft2(cps, workers=-1)).astype(np.float32)
        r -= r.min()
        peak = r.max()
        if peak > 0:
            r /= peak

        n = self._size
        py, px = np.unravel_index(int(np.argmax(r)), r.shape)
        x0, x1 = (px - 1) % n, (px + 1) % n
        y0, y1 = (py - 1) % n, (py + 1) % n
        # 3x3 weighted sub-pixel peak, 0.7071 corner weights (fftalign.jsh:89-100)
        f00 = 0.7071 * r[y0, x0]
        f01 = r[y0, px]
        f02 = 0.7071 * r[y0, x1]
        f10 = r[py, x0]
        f12 = r[py, x1]
        f20 = 0.7071 * r[y1, x0]
        f21 = r[y1, px]
        f22 = 0.7071 * r[y1, x1]
        dx = px - f00 - f10 - f20 + f02 + f12 + f22
        dy = py - f00 - f01 - f02 + f20 + f21 + f22
        if dx >= n / 2:
            dx -= n
        if dy >= n / 2:
            dy -= n
        return float(dx), float(dy)
