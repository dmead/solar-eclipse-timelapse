"""Make the merged corona presentable — port of corona-stretch.js.

INPUT IS corona_hdr.xisf, THE UNFLATTENED MERGE. Running this on corona_flat
instead produces a false-colour mess: the flatten has already divided out the
radial falloff, so the disc interior and the outer field sit at similar levels
and asinh amplifies all of it — including the residual inside the Moon — into
saturation. The flatten and this stretch are two answers to the same problem and
must not be stacked. Use the flatten for a structure-only map; use this for a
picture.

`corona_hdr.xisf` is geometrically correct but tonally useless: the chromosphere
sits near 1.0 and the outer corona near 0.001, seven stops down, and any gamma
that lifts the outer corona washes the inner corona to white.

Three steps, in order.

1. SKY PEDESTAL. Totality sky is not black, it is dusk. That pedestal adds to
   every pixel, so it flattens contrast everywhere and, being additive, breaks
   the stretch below: asinh applied to signal+pedestal compresses the signal by
   whatever the pedestal already used up.

2. ASINH STRETCH. Logarithmic for large values, linear for small, so the faint
   outer corona is lifted hard while the inner corona is compressed gently rather
   than clipped. One gain derived from luminance is applied to all three
   channels, so colour ratios survive — a per-channel curve would desaturate the
   bright inner region toward white.

3. LOCAL CONTRAST. Streamers are low-contrast structure on a smooth gradient.
   Subtracting a blurred copy and adding back a fraction raises them without
   touching overall brightness.
"""

import argparse
import json
import math
import os
import time

import numpy as np

from .imgio import read_xisf, stack_channels, write_xisf

# Annulus for the sky pedestal, in Moon radii, and the percentile taken in it.
SKY_R_INNER, SKY_R_OUTER = 3.2, 3.9
SKY_PERCENTILE = 0.10

ASINH_BETA = 120.0
INNER_TARGET = 0.72

# Two local-contrast scales: a broad one for the streamers, a tight one for
# structure near the limb.
LC_RADIUS, LC_AMOUNT = 20, 0.30
LC2_RADIUS, LC2_AMOUNT = 110, 0.85

__all__ = ["stretch_corona", "box_blur", "asinh_solve", "starlet_contrast",
           "STARLET_BIASES"]

LUMA = (0.25, 0.60, 0.15)

# Per-layer biases for the starlet finish, coarsest-relevant first is NOT the
# order — index j is the a-trous detail layer at scale ~2^j px.
#
#   j=0,1   1-2 px    stack noise and the drizzle grid — held down
#   j=2,3   4-8 px    fine structure at the limb
#   j=4,5   16-32 px  what the r=20 box pass was reaching for
#   j=6,7   64-128 px the streamers themselves, what r=110 was reaching for
#
# MEASURED WORSE THAN THE BOX-BLUR FINISH, 2026-08-16. Kept as an option and
# documented so it is not re-attempted from scratch.
#
# The theory was sound — differencing two box blurs cannot separate scales, so
# lifting streamers also lifts noise and the smooth mid-field, which is
# STATE.md's diagnosis of why the result "reads flat". The practice was not.
# Mean |grad| over the lit field, against the box-blur baseline:
#
#   box blur (verified)                      0.00384   100%
#   coarse-weighted biases                   0.00214    56%
#   mid 8-32 px (streamer filaments)         0.00233    61%
#   mid, harder                              0.00249    65%
#   fine+mid                                 0.00282    74%
#
# Every schedule LOST local structure. Note the trap that hid it at first: the
# starlet versions score HIGHER on standard deviation and p5-p95 spread, because
# they widen the global tonal range. Those are the wrong metrics — the complaint
# is missing structure, so |grad| is the one that answers it.
#
# Prime suspect is the Lab round trip rather than the starlet itself:
# `rgb_to_lab01` decodes its input as sRGB-encoded, which is right for
# lunation's stretched lunar data but applies a nonlinear compression here that
# flattens gradients before the layers are even built. Worth retrying on linear
# luminance, or applying the starlet per channel, before concluding multiscale
# is the wrong tool.
STARLET_BIASES = [0.00, 0.03, 0.10, 0.16, 0.12, 0.05, 0.00, 0.00]


def _annulus_mask(shape, cx, cy, r_in, r_out):
    """Every second pixel inside the annulus, matching the original's stride."""
    H, W = shape
    ys = np.arange(0, H, 2)
    xs = np.arange(0, W, 2)
    dy = (ys - cy)[:, None]
    dx = (xs - cx)[None, :]
    d2 = dx * dx + dy * dy
    return ys, xs, (d2 >= r_in * r_in) & (d2 <= r_out * r_out)


def annulus_median(planes, cx, cy, r_in, r_out, weights=None):
    """Median over an annulus, of one plane or of luminance across three.

    Robust to prominences and stray streamers, which a mean is not.
    """
    ys, xs, m = _annulus_mask(planes[0].shape, cx, cy, r_in, r_out)
    if weights is None:
        vals = planes[0][np.ix_(ys, xs)][m]
    else:
        vals = sum(w * p[np.ix_(ys, xs)] for w, p in zip(weights, planes))[m]
    if vals.size < 100:
        return 0.0
    return float(np.sort(vals)[vals.size // 2])


def asinh_solve(ref, target, beta=ASINH_BETA):
    """Scale k such that asinh(ref*k*beta)/asinh(beta) == target."""
    denom = math.asinh(beta)
    lo, hi = 1e-3, 1e9
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if math.asinh(ref * mid * beta) / denom < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def _box1d(a, r, axis):
    """Mean over a window clipped at the borders, along one axis."""
    n = a.shape[axis]
    c = np.cumsum(a, axis=axis, dtype=np.float64)
    zero = np.zeros_like(np.take(c, [0], axis=axis))
    c = np.concatenate([zero, c], axis=axis)
    i = np.arange(n)
    lo = np.maximum(0, i - r)
    hi = np.minimum(n - 1, i + r)
    cnt = (hi - lo + 1).astype(np.float64)
    top = np.take(c, hi + 1, axis=axis)
    bot = np.take(c, lo, axis=axis)
    shape = [1] * a.ndim
    shape[axis] = n
    return (top - bot) / cnt.reshape(shape)


def box_blur(src, r):
    """Separable box blur, two passes — close enough to Gaussian for local
    contrast and far cheaper on an 8 Mpx frame."""
    out = src
    for _ in range(2):
        out = _box1d(_box1d(out, r, axis=1), r, axis=0)
    return out.astype(np.float32)


def starlet_contrast(ch, biases=None, log=print):
    """Multiscale local contrast on L* alone — the alternative to step 3.

    Two changes from the box-blur finish, both aimed at the same complaint.

    Scale separation: an a-trous starlet splits the image into layers that do
    not overlap, so the streamers can be lifted without dragging the noise and
    the smooth mid-field up with them. Differencing two box blurs cannot do
    that — every difference contains all the finer scales too.

    Luminance only: the work happens on L* in CIELab and a*/b* are carried
    through untouched, so raising contrast cannot shift colour. The box-blur
    version runs per channel, which is why lifting it hard tends to push the
    inner corona toward white — the same failure the asinh step already avoids
    by deriving one gain from luminance.
    """
    from .vendor.finish.primitives import (lab01_to_rgb, rgb_to_lab01,
                                            starlet_sharpen)

    biases = STARLET_BIASES if biases is None else biases
    rgb = np.stack(ch, axis=-1)
    L, a, b = rgb_to_lab01(rgb)
    L2 = starlet_sharpen(L, list(biases))
    out = lab01_to_rgb(L2, a, b)
    log(f"  starlet contrast on L*, {len(biases)} layers, biases {biases}")
    return [np.clip(out[:, :, c], 0.0, 1.0).astype(np.float32) for c in range(3)]


def stretch_corona(in_path, out_path, moon_path, log=print,
                   contrast="boxblur", inner_target=None, biases=None):
    t0 = time.time()
    with open(moon_path, encoding="utf-8-sig") as f:
        moon = json.load(f)

    img = read_xisf(in_path)
    H, W = img.shape[:2]
    # Clamp to [0, 1] on read, as PixInsight's Image does for a float image.
    #
    # Not merely for compatibility. The HDR merge legitimately leaves channel 0
    # slightly below zero out in the sky — its affine pedestal fit has no reason
    # to land exactly on zero — and measured here the sky annulus is 100%
    # negative in R while G and B are 0%. Without the clamp the step-1
    # percentile comes out at -5.76e-4, and subtracting a NEGATIVE pedestal adds
    # light, which is the exact opposite of what step 1 exists to do: the sky
    # pedestal is additive dusk and can only be removed. Clamping makes the
    # measured pedestal non-negative by construction.
    ch = [np.clip(img[:, :, c], 0.0, 1.0).astype(np.float32) for c in range(3)]

    # The sidecar centre is in the UNCROPPED frame when it came from combine;
    # corona-hdr trimmed to common coverage and records a trim offset when it
    # applies.
    cx = moon["cx"] - moon.get("trimX", 0)
    cy = moon["cy"] - moon.get("trimY", 0)
    R = moon["radius"]
    log(f"stretch {W}x{H} moon ({cx:.0f}, {cy:.0f}) r={R:.0f}")

    # ---- 1. sky pedestal ----
    ys, xs, m = _annulus_mask((H, W), cx, cy, SKY_R_INNER * R, SKY_R_OUTER * R)
    for c in range(3):
        vals = ch[c][np.ix_(ys, xs)][m]
        sky = 0.0
        if vals.size > 200:
            sky = float(np.sort(vals)[int(vals.size * SKY_PERCENTILE)])
        log(f"  channel {c} sky pedestal {sky:.3e} from {vals.size} samples")
        ch[c] = np.maximum(ch[c] - sky, 0.0)

    # ---- 1b. white balance on the corona itself ----
    # The channels carry different sky pedestals and different sensor response,
    # and subtracting unequal pedestals leaves a cast of its own — the first
    # result had a distinctly green limb. The corona is very nearly white in
    # reality, so equalising channel medians over real corona is defensible and
    # costs nothing in structure.
    med = [annulus_median([ch[c]], cx, cy, 1.15 * R, 2.10 * R) for c in range(3)]
    mref = sum(med) / 3
    for c in range(3):
        if not med[c] > 0:
            continue
        g = mref / med[c]
        log(f"  wb channel {c} median {med[c]:.3e} gain {g:.3f}")
        ch[c] *= g

    # ---- 2. asinh stretch, luminance-driven ----
    ref = annulus_median(ch, cx, cy, 1.05 * R, 1.25 * R, weights=LUMA)
    if not ref > 0:
        raise ValueError("inner corona reference measured as zero")
    target = INNER_TARGET if inner_target is None else inner_target
    k = asinh_solve(ref, target)
    log(f"  inner corona ref {ref:.3e}, scale {k:.1f}, beta {ASINH_BETA}, "
        f"target {target}")

    denom = math.asinh(ASINH_BETA)
    L = LUMA[0] * ch[0] + LUMA[1] * ch[1] + LUMA[2] * ch[2]
    pos = L > 0
    gain = np.zeros_like(L)
    gain[pos] = np.arcsinh(L[pos] * k * ASINH_BETA) / denom / L[pos]
    for c in range(3):
        v = ch[c] * gain
        ch[c] = np.where(pos, np.minimum(v, 1.0), ch[c])

    # ---- 2b. blank the Moon ----
    # Whatever survives inside the lunar disc is scattered light and stack
    # residual, not corona. asinh lifts it to a visible grey plate and the eye
    # reads that as fog over the whole image.
    r_blank, r_feather = 0.985 * R, 0.045 * R
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.hypot(xx - cx, yy - cy)
    inside = d < r_blank
    f = np.where(d > r_blank - r_feather,
                 (d - (r_blank - r_feather)) / r_feather, 0.0)
    for c in range(3):
        ch[c] = np.where(inside, ch[c] * f, ch[c]).astype(np.float32)
    log(f"  moon interior blanked inside r={r_blank:.0f}")

    # ---- 3. local contrast ----
    if contrast == "starlet":
        ch = starlet_contrast(ch, biases, log=log)
    else:
        for rad, amt in ((LC2_RADIUS, LC2_AMOUNT), (LC_RADIUS, LC_AMOUNT)):
            if not amt > 0:
                continue
            for c in range(3):
                blur = box_blur(ch[c], rad)
                ch[c] = np.clip(ch[c] + amt * (ch[c] - blur),
                                0.0, 1.0).astype(np.float32)
            log(f"  local contrast r={rad} amount {amt}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    write_xisf(out_path, stack_channels(*ch))
    log(f"  saved {out_path} in {time.time() - t0:.0f} s")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in-path", default="S:/solar-eclipse/out/final/corona_hdr.xisf")
    ap.add_argument("--out", default="S:/solar-eclipse/out/final/corona_final.xisf")
    ap.add_argument("--moon", default="S:/solar-eclipse/out/final/corona_flat_moon.json")
    ap.add_argument("--contrast", default="boxblur", choices=["boxblur", "starlet"],
                    help="step 3: the verified two-scale box blur, or the "
                         "multiscale starlet finish on L*")
    ap.add_argument("--inner-target", type=float, default=None,
                    help=f"asinh target for the inner corona (default {INNER_TARGET})")
    args = ap.parse_args(argv)
    stretch_corona(args.in_path, args.out, args.moon,
                   contrast=args.contrast, inner_target=args.inner_target)


if __name__ == "__main__":
    main()
