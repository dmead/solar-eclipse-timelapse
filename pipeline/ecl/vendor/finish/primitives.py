"""Finishing-chain numeric primitives.

Replacements for the PixInsight processes lunar-finish.js invokes:
MTF/HistogramTransformation, Convolution (gaussian), starlet wavelets
(MultiscaleLinearTransform), Richardson-Lucy deconvolution with wavelet
regularization + dark deringing, CurvesTransformation (monotone PCHIP —
better-than decision: no overshoot, unlike interpolating splines), and
sRGB<->CIELab (D65; PI's RGBWS quirks deliberately not replicated).
"""

import cv2
import numpy as np
import scipy.fft as sfft
from scipy.interpolate import PchipInterpolator


# ---------------------------------------------------------------- tone

def mtf(m: float, x: np.ndarray | float):
    """PixInsight midtones transfer function."""
    x = np.asarray(x, dtype=np.float32)
    return ((m - 1) * x) / ((2 * m - 1) * x - m)


def mtf_for(x: float, y: float) -> float:
    """Midtones balance m such that mtf(m, x) = y (lunar-finish.js:130-133)."""
    return x * (y - 1) / ((2 * y - 1) * x - y)


def histogram_transform(img: np.ndarray, shadow: float, mid: float,
                        high: float = 1.0) -> np.ndarray:
    """HistogramTransformation: clip to [shadow, high], then MTF(mid)."""
    x = np.clip((img - shadow) / max(high - shadow, 1e-6), 0.0, 1.0)
    return mtf(mid, x).astype(np.float32)


def curve(img: np.ndarray, points: list[list[float]]) -> np.ndarray:
    """Monotone curve through the given [[x,y],...] control points."""
    pts = sorted(points)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    f = PchipInterpolator(xs, ys)
    return np.clip(f(np.clip(img, xs[0], xs[-1])), 0, 1).astype(np.float32)


def gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(np.asarray(img, dtype=np.float32), (0, 0), sigma)


def pi_convolution(img: np.ndarray, sigma: float) -> np.ndarray:
    """PI Convolution (Parametric, shape 2.0) equivalent. PI's parametric
    kernels are exp(-(r/sigma)^shape) — a true Gaussian of sigma/sqrt(2)
    (same convention as its deconvolution PSF, verified 2026-07-16), so a
    'sigma 24' PI blur is narrower than a sigma-24 Gaussian."""
    return gaussian(img, sigma / np.sqrt(2.0))


# ---------------------------------------------------------------- starlet

_B3 = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0


def _b3_blur(img: np.ndarray, level: int) -> np.ndarray:
    """Separable a-trous B3-spline smoothing at dyadic scale 2^level."""
    k = np.zeros(4 * 2**level + 1, dtype=np.float32)
    k[:: 2**level] = _B3
    return cv2.sepFilter2D(img, cv2.CV_32F, k, k,
                           borderType=cv2.BORDER_REFLECT)


def starlet(img: np.ndarray, n_layers: int) -> list[np.ndarray]:
    """A-trous starlet: n_layers detail layers + residual (last)."""
    layers = []
    c = np.asarray(img, dtype=np.float32)
    for j in range(n_layers):
        s = _b3_blur(c, j)
        layers.append(c - s)
        c = s
    layers.append(c)
    return layers


# PI's MLT bias k maps to a layer multiplier of ~(1 + PI_BIAS_GAIN*k), not
# (1+k): calibrated 2026-07-16 against PI's measured MLT stage gains on the
# 2026-06-30 L (PI x1.83 at the k=0.12/0.13 layers where (1+k) gives x1.06).
PI_BIAS_GAIN = 12.0


def starlet_sharpen(img: np.ndarray, biases: list[float],
                    deringing: bool = True, dering_amt: float = 0.01
                    ) -> np.ndarray:
    """MLT starlet bias sharpening:
    out = residual + sum (1 + PI_BIAS_GAIN*bias_k) * w_k.

    Deringing: dense lunar texture punishes per-layer excursion limiting
    (a tanh knee and a min/max clamp were both tried and crushed detail,
    2026-07-16); ringing control is left to the final [0,1] clip, the
    smooth disk mask, and visual review. `deringing` is accepted for
    config compatibility.
    """
    del deringing, dering_amt
    layers = starlet(img, len(biases))
    out = layers[-1].copy()
    for w, b in zip(layers[:-1], biases):
        out += w * (1.0 + PI_BIAS_GAIN * b)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------- deconvolve

def rl_deconvolve(img: np.ndarray, psf_sigma: float, iterations: int,
                  dark_dering: float = 0.1, reg_layers: int = 2,
                  reg_sigma: float = 2.0) -> np.ndarray:
    """Richardson-Lucy with a parametric Gaussian PSF, wavelet-regularized
    correction, and dark deringing.

    - regularization: the multiplicative correction's finest starlet layers
      are SOFT-THRESHOLDED at reg_sigma x their robust noise scale — noise
      never amplifies, while real edges (far above sigma) pass at full
      strength. (Blanket attenuation was tried first and halves RL's
      sharpening on high-SNR lunar data — 2026-07-16. reg_sigma default
      raised 1.0 -> 2.0: at 1.0, ~1-sigma drizzle-correlated noise passes
      the threshold every iteration and compounds into a 2px grid with
      square a-trous rings on single-L-stack nights — 2026-07-24.)
    - dark deringing: the update is damped where the image is darker than
      dark_dering x its local (PSF-scale) mean — undershoot against black
      sky is where RL ringing lives on lunar limbs.
    """
    # iteration-equivalence calibration: PI's RL converges ~3x faster per
    # nominal iteration than this damped/accelerated implementation (fitted
    # against PI's per-band deconv gains, 2026-07-16) — configs are a shared
    # contract, so "iterations: 25" must mean PI-25 in both pipelines.
    iterations = round(iterations * 3.0)
    x = np.clip(np.asarray(img, dtype=np.float32), 1e-6, 1.0)
    h, w = x.shape
    # PSF via FFT (periodic conv is fine: the sky border is black).
    # PI's parametric PSF is exp(-(r/sigma)^shape) — NOT exp(-r^2/2s^2):
    # at shape 2 that is a Gaussian of sigma/sqrt(2). Verified 2026-07-16 by
    # fitting PI's per-band deconv gains to inverse-OTF asymptotes
    # (1.03/1.29/2.96/13.5 measured vs 1.04/1.32/3.2/40 asymptotic).
    fh = sfft.next_fast_len(h)
    fw = sfft.next_fast_len(w)
    yy = np.minimum(np.arange(fh), fh - np.arange(fh))[:, None]
    xx = np.minimum(np.arange(fw), fw - np.arange(fw))[None, :]
    r2 = (yy**2 + xx**2).astype(np.float32)
    psf = np.exp(-(r2 / psf_sigma**2)).astype(np.float32)
    psf /= psf.sum()
    otf = sfft.rfft2(psf)
    otf_conj = np.conj(otf)

    def conv(a, kernel_f):
        pad = np.zeros((fh, fw), dtype=np.float32)
        pad[:h, :w] = a
        return sfft.irfft2(sfft.rfft2(pad) * kernel_f, s=(fh, fw)
                           )[:h, :w].astype(np.float32)

    est = x.copy()
    local = gaussian(x, 2.0 * psf_sigma)  # deringing support
    dering_w = np.clip(x / np.maximum(dark_dering * np.maximum(local, 1e-6),
                                      1e-6), 0.0, 1.0) if dark_dering > 0 \
        else np.ones_like(x)
    # Biggs-Andrews vector acceleration (deconvlucy form): PI's RL reaches
    # near-asymptotic fine-scale gain in ~25 iterations; plain RL needs
    # hundreds there.
    g1 = g2 = x_prev = None
    for _ in range(iterations):
        if g1 is not None and g2 is not None:
            num = float((g1 * g2).sum())
            den = float((g2 * g2).sum())
            alpha = min(max(num / max(den, 1e-12), 0.0), 0.99)
            y = np.clip(est + alpha * (est - x_prev), 1e-8, 1.0)
        else:
            y = est
        denom = np.maximum(conv(y, otf), 1e-6)
        ratio = x / denom
        corr = conv(ratio, otf_conj)
        # wavelet regularization: soft-threshold the correction's fine layers
        if reg_layers > 0:
            lay = starlet(corr, reg_layers)
            corr = lay[-1]
            for wl in lay[:-1]:
                s = 1.4826 * float(np.median(np.abs(wl - np.median(wl))))
                corr = corr + np.sign(wl) * np.maximum(
                    np.abs(wl) - reg_sigma * s, 0.0)
        upd = y * corr
        new = y + (upd - y) * dering_w
        np.clip(new, 0.0, 1.0, out=new)
        g2 = g1
        g1 = new - y
        x_prev = est
        est = new
    return est


# ---------------------------------------------------------------- CIELab

_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ).astype(np.float32)
_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)  # D65


def _f_lab(t):
    d = 6 / 29
    return np.where(t > d**3, np.cbrt(t), t / (3 * d * d) + 4 / 29)


def _f_lab_inv(t):
    d = 6 / 29
    return np.where(t > d, t**3, 3 * d * d * (t - 4 / 29))


def _srgb_decode(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb_encode(c):
    return np.where(c <= 0.0031308, 12.92 * c,
                    1.055 * np.power(np.maximum(c, 0), 1 / 2.4) - 0.055)


def rgb_to_lab01(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Working RGB (sRGB-ENCODED floats, as PI's stretched pipeline data)
    -> (L*, a*, b*) each normalized to [0,1] with a*/b* centered at 0.5
    (the layout PI's CIELab channels use). The encode/decode matters:
    assigning a stretched L image as L* only lands PI-scale output levels
    when the RGB side is treated as gamma-encoded (verified vs the PI
    final's disk mean, 2026-07-16)."""
    lin = _srgb_decode(np.clip(rgb, 0, 1))
    xyz = lin @ _M_RGB2XYZ.T / _WHITE
    fx, fy, fz = (_f_lab(xyz[..., i]) for i in range(3))
    L = 116 * fy - 16          # [0,100]
    a = 500 * (fx - fy)        # ~[-128,128]
    b = 200 * (fy - fz)
    return ((L / 100).astype(np.float32),
            (a / 255 + 0.5).astype(np.float32),
            (b / 255 + 0.5).astype(np.float32))


def lab01_to_rgb(L: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    Ls = L * 100
    as_ = (a - 0.5) * 255
    bs = (b - 0.5) * 255
    fy = (Ls + 16) / 116
    fx = fy + as_ / 500
    fz = fy - bs / 200
    xyz = np.stack([_f_lab_inv(fx), _f_lab_inv(fy), _f_lab_inv(fz)],
                   axis=-1) * _WHITE
    lin = np.clip(xyz @ _M_XYZ2RGB.T, 0, 1)
    return np.clip(_srgb_encode(lin), 0, 1).astype(np.float32)
