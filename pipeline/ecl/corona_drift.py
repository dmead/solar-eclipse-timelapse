"""Measure how fast the Moon slides across the corona — port of corona-drift.js.

The Moon and the corona are not the same reference frame. The corona is fixed to
the Sun; the Moon crosses it at roughly its own synodic rate. Over one 60 s
capture that is a handful of pixels, but across the whole exposure ladder it is
tens of pixels — enough to double the streamers in an HDR merge, or to double the
limb, depending on which feature the registration happened to lock onto.

Rather than trust an ephemeris and an assumed image scale, this measures the rate
from the data: take two stacks of the SAME exposure level at different times, put
them in a common Moon-centred frame, mask out everything but the outer corona,
and cross-correlate. Whatever shift remains is the Moon-vs-Sun slip over the
known interval.

The correlation runs on the radially FLATTENED image. Without that the profile
itself dominates and every candidate shift near zero scores about the same,
because a smooth radial gradient correlates with itself.
"""

import argparse
import json
import math
import os
import time

import numpy as np

from .imgio import read_xisf

# Annulus for the correlation, in units of the Moon's radius. The inner bound
# clears the limb and the chromosphere. The outer bound is deliberately close in:
# a first attempt reaching to 3.2 R measured a drift three times the physical
# ceiling, because that far out the corona is faint and the correlation locked
# onto the sensor's own flat-field signature, which does not move between frames
# and so reported the Moon's full frame motion as if it were differential.
R_INNER = 1.10
R_OUTER = 2.00

# Physical ceiling on the differential rate. Converting the synodic rate through
# the MEASURED Moon radius gives pixels per second without needing the focal
# length. Anything above this is a failed correlation, not a fast Moon.
SYNODIC_ARCSEC_PER_S = 0.51
MOON_RADIUS_ARCSEC = 990.0
DRIFT_SAFETY = 1.4

SEARCH_COARSE = 14
DECIM = 8
SEARCH_FINE = 6
MIN_OVERLAP = 500

__all__ = ["measure_drift", "flatten_annulus", "pick_drift_pair"]


def flatten_annulus(g, cx, cy, r_in, r_out):
    """Divide out the azimuthally averaged radial profile inside an annulus.

    Returns the fractional excess over the profile, plus the annulus mask. What
    is left is what the correlation should lock onto: the streamers, not the
    falloff.
    """
    H, W = g.shape
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.hypot(xx - cx, yy - cy)
    inside = (d >= r_in) & (d <= r_out)

    r_max = int(math.ceil(r_out))
    ri = np.floor(d + 0.5).astype(np.int64)          # JS Math.round
    np.clip(ri, 0, r_max, out=ri)
    sel = ri[inside]
    cnt = np.bincount(sel, minlength=r_max + 1)
    tot = np.bincount(sel, weights=g[inside].astype(np.float64),
                      minlength=r_max + 1)
    prof = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)

    p = prof[ri]
    ok = inside & (p > 0)
    img = np.zeros_like(g, dtype=np.float32)
    img[ok] = (g[ok] / p[ok] - 1.0).astype(np.float32)
    return img, ok


def decimate(img, mask, k):
    """Block mean over masked pixels, keeping a cell only if it is mostly valid."""
    h, w = img.shape
    hh, ww = h // k, w // k
    im = img[:hh * k, :ww * k].reshape(hh, k, ww, k)
    mk = mask[:hh * k, :ww * k].reshape(hh, k, ww, k)
    c = mk.sum(axis=(1, 3))
    s = np.where(mk, im, 0).sum(axis=(1, 3))
    keep = c > (k * k >> 1)
    out = np.zeros((hh, ww), np.float32)
    np.divide(s, np.maximum(c, 1), out=out, where=keep)
    return out, keep


def ncc(a_img, a_mask, b_img, b_mask, dx, dy):
    """Masked normalized cross-correlation of b shifted by (dx, dy) against a."""
    h, w = a_img.shape
    y0, y1 = max(0, -dy), min(h, h - dy)
    x0, x1 = max(0, -dx), min(w, w - dx)
    if y1 <= y0 or x1 <= x0:
        return -2.0
    A = a_img[y0:y1, x0:x1]
    B = b_img[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    M = a_mask[y0:y1, x0:x1] & b_mask[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    n = int(M.sum())
    if n < MIN_OVERLAP:
        return -2.0
    u = A[M].astype(np.float64)
    v = B[M].astype(np.float64)
    ma, mb = u.mean(), v.mean()
    ca = float((u * u).sum()) - n * ma * ma
    cb = float((v * v).sum()) - n * mb * mb
    if ca <= 0 or cb <= 0:
        return -2.0
    return float((float((u * v).sum()) - n * ma * mb) / math.sqrt(ca * cb))


def _search(a, am, b, bm, cx, cy, win):
    best = (cx, cy, -2.0)
    for dy in range(cy - win, cy + win + 1):
        for dx in range(cx - win, cx + win + 1):
            s = ncc(a, am, b, bm, dx, dy)
            if s > best[2]:
                best = (dx, dy, s)
    return best


def translate_image(src, dx, dy):
    """Bilinear translate, leaving zero outside the source.

    Replicating the edge pixel instead would smear a bright border inward and it
    would pass the signal test as if it were data.
    """
    H, W = src.shape
    ys = np.arange(H) - dy
    xs = np.arange(W) - dx
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    fy = (ys - y0).astype(np.float32)
    fx = (xs - x0).astype(np.float32)
    y1, x1 = y0 + 1, x0 + 1

    vy = (y1 >= 0) & (y0 < H)
    vx = (x1 >= 0) & (x0 < W)
    # Clamp the fractional weight where the pair straddles the border, matching
    # the original's `if (y0 < 0) { y0 = 0; fy = 0; }`.
    fy = np.where(y0 < 0, 0.0, fy)
    fx = np.where(x0 < 0, 0.0, fx)
    y0c, y1c = np.clip(y0, 0, H - 1), np.clip(y1, 0, H - 1)
    x0c, x1c = np.clip(x0, 0, W - 1), np.clip(x1, 0, W - 1)

    dst = np.zeros_like(src)
    Y0, Y1 = y0c[vy][:, None], y1c[vy][:, None]
    X0, X1 = x0c[vx][None, :], x1c[vx][None, :]
    FY, FX = fy[vy][:, None], fx[vx][None, :]
    u = src[Y0, X0] + (src[Y0, X1] - src[Y0, X0]) * FX
    v = src[Y1, X0] + (src[Y1, X1] - src[Y1, X0]) * FX
    dst[np.ix_(np.nonzero(vy)[0], np.nonzero(vx)[0])] = u + (v - u) * FY
    return dst


def _plane(path):
    img = read_xisf(path)
    return img[:, :, 1] if img.ndim == 3 else img


def measure_drift(a, b, log=print):
    """`a` and `b` are dicts of {path, cx, cy, r, t} plus optional {dx, dy}."""
    t0 = time.time()
    dt = b["t"] - a["t"]
    if not abs(dt) > 1:
        raise ValueError(f"need a meaningful time baseline, got dt={dt}s")

    A, B = _plane(a["path"]), _plane(b["path"])
    if A.shape != B.shape:
        raise ValueError("geometry mismatch between the two stacks")

    # Common Moon-centred frame: shift B so its Moon sits on A's. Any residual
    # corona shift after this is the differential motion. The offset comes from
    # the registration when available, since differencing two independent Moon
    # fits is exactly what proved unreliable.
    if a.get("dx") is not None and b.get("dx") is not None:
        mx, my = b["dx"] - a["dx"], b["dy"] - a["dy"]
    else:
        mx, my = a["cx"] - b["cx"], a["cy"] - b["cy"]
    b_aligned = translate_image(B, mx, my)
    log(f"  moon-registered B onto A: ({mx:.2f}, {my:.2f}) px")

    r_in, r_out = a["r"] * R_INNER, a["r"] * R_OUTER
    fa, ma = flatten_annulus(A, a["cx"], a["cy"], r_in, r_out)
    fb, mb = flatten_annulus(b_aligned, a["cx"], a["cy"], r_in, r_out)

    da, dam = decimate(fa, ma, DECIM)
    db, dbm = decimate(fb, mb, DECIM)
    cdx, cdy, cs = _search(da, dam, db, dbm, 0, 0, SEARCH_COARSE)
    log(f"  coarse best ({cdx * DECIM}, {cdy * DECIM}) ncc={cs:.5f}")

    fdx, fdy, fs = _search(fa, ma, fb, mb, cdx * DECIM, cdy * DECIM, SEARCH_FINE)
    log(f"  refined best ({fdx:.2f}, {fdy:.2f}) ncc={fs:.5f}")

    # The refined shift is how far B's corona sits from A's after Moon
    # registration, so the corona moves by -shift per dt in the Moon frame.
    vx, vy = -fdx / dt, -fdy / dt
    speed = math.hypot(vx, vy)
    log(f"  baseline {dt:.1f} s")
    log(f"  differential drift {speed:.4f} px/s  ({vx:.4f}, {vy:.4f})")
    log(f"  => {speed * 60:.1f} px per 60 s capture")

    # Physics gate. A rate above the synodic ceiling means the correlation found
    # something that is not corona; hand back zero so the caller falls back to
    # rigid registration rather than shifting every level by a fabricated amount.
    ceiling = DRIFT_SAFETY * SYNODIC_ARCSEC_PER_S * a["r"] / MOON_RADIUS_ARCSEC
    accepted = speed <= ceiling
    log(f"  ceiling {ceiling:.4f} px/s (moon r={a['r']:.1f} px) -> "
        f"{'accepted' if accepted else 'REJECTED'}")
    if not accepted:
        log("  *** measured rate exceeds what the Moon can do; reporting zero drift")
        vx = vy = 0.0

    log(f"  measured in {time.time() - t0:.0f} s")
    return {
        "dtSeconds": dt,
        "shiftPx": {"dx": fdx, "dy": fdy},
        "driftPxPerSec": {"x": vx, "y": vy, "speed": math.hypot(vx, vy)},
        "measuredSpeed": speed,
        "ceiling": ceiling,
        "accepted": accepted,
        "ncc": fs,
        "annulus": {"rInner": r_in, "rOuter": r_out},
    }


def pick_drift_pair(images):
    """Two stacks of the same exposure level with the longest time baseline."""
    by_level = {}
    for im in images:
        by_level.setdefault(im["level"], []).append(im)
    best = None
    for group in by_level.values():
        if len(group) < 2:
            continue
        s = sorted(group, key=lambda i: i["t"])
        dt = abs(s[-1]["t"] - s[0]["t"])
        if best is None or dt > best["dt"]:
            best = {"a": s[0], "b": s[-1], "dt": dt}
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True,
                    help="corona_drift.json with {a, b, out}")
    args = ap.parse_args(argv)
    with open(args.config, encoding="utf-8-sig") as f:
        cfg = json.load(f)
    out = measure_drift(cfg["a"], cfg["b"])
    os.makedirs(os.path.dirname(os.path.abspath(cfg["out"])), exist_ok=True)
    with open(cfg["out"], "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"wrote {cfg['out']}")


if __name__ == "__main__":
    main()
