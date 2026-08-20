"""Merge the exposure ladder and divide out the radial profile — port of
corona-hdr.js.

`HDRComposition` is deliberately not used, in the original or here: the merge is a
handful of arithmetic decisions and doing them explicitly keeps them inspectable.

REFERENCE FRAME. The Moon and the corona move relative to each other, so no
single rigid registration holds both. The output is defined as the Moon frame at
one instant — the shortest exposure's capture time, since that is the level
carrying the limb, chromosphere and prominences. Every other level is placed so
its CORONA lands where the corona was at that instant: Moon registration first,
then the measured differential drift taken back out.
"""

import argparse
import json
import math
import os
import time

import numpy as np

from .imgio import read_xisf, stack_channels, write_xisf

# Saturation roll-off and signal roll-on, in linear level units.
SAT_LO, SAT_HI = 0.70, 0.94
SIG_LO, SIG_HI = 0.0008, 0.0060

# Fitting windows for the level-to-level affine relation.
FIT_SHORT_LO, FIT_SHORT_HI = 0.0020, 0.90
FIT_LONG_LO, FIT_LONG_HI = 0.0050, 0.85
FIT_MIN_SPAN = 3.0
FIT_STRIDE = 37          # coprime with the row length: samples the whole frame

# Fraction of the radial profile put back after dividing it out.
PROFILE_RESTORE = 0.45

# A level whose Moon has moved is excluded from inside this feather.
LIMB_FEATHER_PX = 90.0
LIMB_TOL_PX = 1.5

# A radius sampled by fewer pixels than this has no meaningful median.
MIN_RADIAL_SAMPLES = 2000

__all__ = ["merge_hdr", "fit_affine", "flatten", "ramp"]


def ramp(v, lo, hi):
    """Smooth 0->1 ramp, so levels enter and leave the blend gradually."""
    t = np.clip((np.asarray(v, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def translate_image(src, dx, dy):
    """Bilinear whole-image translation, leaving ZERO outside the source.

    Registration between exposure levels is pure translation: the mount tracked
    and the frames are seconds apart. Clamping to the edge pixel instead smeared
    a bright strip down two sides of the merge — and because the strip is bright
    it passed the signal test and was blended in as if it were real data. Zero
    fails that test, so those pixels fall back to the unshifted reference.
    """
    H, W = src.shape
    ys, xs = np.arange(H) - dy, np.arange(W) - dx
    y0, x0 = np.floor(ys).astype(np.int64), np.floor(xs).astype(np.int64)
    fy, fx = (ys - y0).astype(np.float32), (xs - x0).astype(np.float32)
    vy, vx = (y0 >= 0) & (y0 + 1 < H), (x0 >= 0) & (x0 + 1 < W)
    dst = np.zeros_like(src)
    if not vy.any() or not vx.any():
        return dst
    Y0 = y0[vy][:, None]
    X0 = x0[vx][None, :]
    FY, FX = fy[vy][:, None], fx[vx][None, :]
    a = src[Y0, X0] + (src[Y0, X0 + 1] - src[Y0, X0]) * FX
    b = src[Y0 + 1, X0] + (src[Y0 + 1, X0 + 1] - src[Y0 + 1, X0]) * FX
    dst[np.ix_(np.nonzero(vy)[0], np.nonzero(vx)[0])] = a + (b - a) * FY
    return dst


def fit_affine(short_c, long_c):
    """Least-squares `long = a*short + b` over pixels well exposed in both.

    The intercept is the point: it absorbs the difference in pedestal between the
    two exposures. Without it the fit is forced through the origin and can only
    be right over a narrow band of brightness — measured on this data, a ratio
    fitted near the limb overpredicts the longer exposure by 315% out at 2.5 Moon
    radii, and the sign change on the way printed a ring at 1.47 R. An affine fit
    holds to about 10% over the same span.

    Outliers are trimmed once against the first fit, so prominences and the
    chromosphere — which sit at the top of the window and are not part of the
    smooth corona relation — cannot tilt the line.
    """
    a_s = short_c.reshape(-1)[::FIT_STRIDE]
    b_s = long_c.reshape(-1)[::FIT_STRIDE]
    sel = ((a_s > FIT_SHORT_LO) & (a_s < FIT_SHORT_HI)
           & (b_s > FIT_LONG_LO) & (b_s < FIT_LONG_HI))
    xs, ys = a_s[sel].astype(np.float64), b_s[sel].astype(np.float64)
    if xs.size == 0:
        return {"a": 0.0, "b": 0.0, "n": 0, "span": 0.0}
    xlo, xhi = float(xs.min()), float(xs.max())
    if xs.size < 100 or not (xhi > xlo * FIT_MIN_SPAN):
        return {"a": 0.0, "b": 0.0, "n": int(xs.size),
                "span": (xhi / xlo if xlo > 0 else 0.0)}

    def solve(x, y):
        m = x.size
        sx, sy = x.sum(), y.sum()
        d = m * float((x * x).sum()) - sx * sx
        if not abs(d) > 0:
            return 0.0, 0.0
        a = (m * float((x * y).sum()) - sx * sy) / d
        return a, (sy - a * sx) / m

    a, b = solve(xs, ys)
    if not a > 0:
        return {"a": a, "b": b, "n": int(xs.size), "span": xhi / xlo}

    res = np.abs(ys - (a * xs + b))
    cut = 3 * float(np.sort(res)[res.size // 2]) + 1e-9
    keep = res <= cut
    if int(keep.sum()) >= 100:
        a, b = solve(xs[keep], ys[keep])
    return {"a": a, "b": b, "n": int(xs.size), "span": xhi / xlo}


def flatten(rgb, cx, cy, moon_r, log=print):
    """Divide out the corona's radial brightness profile.

    The profile is a per-radius median of luminance, so streamers — local
    excursions — do not pull it up. Inside the limb there is no corona to
    flatten, so the profile is held at its limb value and the disc keeps its own
    darkness.
    """
    H, W = rgb[0].shape
    rmax = int(math.ceil(math.hypot(max(cx, W - cx), max(cy, H - cy))))

    yy, xx = np.mgrid[0:H, 0:W]
    d = np.hypot(xx - cx, yy - cy)
    r = np.floor(d + 0.5).astype(np.int64)          # JS Math.round

    lum = 0.25 * rgb[0] + 0.60 * rgb[1] + 0.15 * rgb[2]
    BINS, LMIN, LMAX = 2048, -20.0, 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        l2 = np.where(lum > 0, np.log2(np.maximum(lum, 1e-300)), LMIN)
    b = np.floor((l2 - LMIN) / (LMAX - LMIN) * (BINS - 1) + 0.5).astype(np.int64)
    np.clip(b, 0, BINS - 1, out=b)

    inb = r <= rmax
    rr, bb = r[inb], b[inb]
    cnt = np.bincount(rr, minlength=rmax + 1)
    hist = np.bincount(rr * BINS + bb, minlength=(rmax + 1) * BINS
                       ).reshape(rmax + 1, BINS)

    # Beyond the largest circle that fits in the frame a radius is sampled only
    # where it clips a corner, so its median comes from a handful of pixels and
    # is not a profile at all. Dividing by it multiplied the corners into a
    # bright band around two edges. Past the last radius with enough support the
    # profile is frozen — the corners then get a constant gain, which is honest:
    # there is no measurement out there to correct with. Scan outward from the
    # limb, since the innermost radii legitimately have almost no pixels.
    r_valid = rmax
    for rr_ in range(max(1, int(math.ceil(moon_r))), rmax + 1):
        if cnt[rr_] < MIN_RADIAL_SAMPLES:
            r_valid = rr_ - 1
            break

    prof = np.zeros(rmax + 1, dtype=np.float64)
    csum = np.cumsum(hist[:r_valid + 1], axis=1)
    half = cnt[:r_valid + 1] >> 1
    for rr_ in range(r_valid + 1):
        if cnt[rr_] == 0:
            prof[rr_] = prof[rr_ - 1] if rr_ > 0 else 1e-6
            continue
        bi = int(np.searchsorted(csum[rr_], half[rr_], side="left"))
        # The original stops at BINS-1 and leaves b one past the bin that
        # crossed the halfway mark, so reproduce that off-by-one exactly.
        bi = min(bi + 1, BINS - 1)
        prof[rr_] = 2.0 ** (LMIN + bi / (BINS - 1) * (LMAX - LMIN))
    prof[r_valid + 1:] = prof[r_valid]
    log(f"  radial profile valid to r={r_valid} of {rmax} px; held constant beyond")

    r_in = max(1, int(np.floor(moon_r * 0.99 + 0.5)))
    if r_in <= rmax:
        prof[:r_in] = prof[r_in]
    # Smooth so pixel-scale noise in the profile does not print as rings.
    K = 9
    pad = np.pad(prof, K, mode="constant")
    win = np.ones(2 * K + 1)
    num = np.convolve(pad, win, mode="valid")
    den = np.convolve(np.pad(np.ones_like(prof), K), win, mode="valid")
    sm = num / den

    ref = sm[r_in] if sm[r_in] > 0 else 1e-6
    rc = np.minimum(r, rmax)
    pr = np.where(sm[rc] > 0, sm[rc], 1e-6)
    # Full division would flatten the corona into a uniform sheet; putting a
    # fraction of the profile back keeps the natural inner-to-outer gradient.
    gain = (ref / pr) * (pr / ref) ** PROFILE_RESTORE
    out = [(c * gain).astype(np.float32) for c in rgb]
    log(f"  flattened about ({cx:.1f}, {cy:.1f}), limb r={moon_r}, "
        f"profile restored {PROFILE_RESTORE}")
    return out


def merge_hdr(cfg, log=print):
    """Merge the ladder, trim to common coverage, flatten, and save."""
    t0 = time.time()
    specs = sorted(cfg["images"], key=lambda s: s["level"])
    ref = specs[0]
    t_ref = ref.get("t", 0.0)
    drift = cfg.get("drift") or {"x": 0.0, "y": 0.0}
    has_drift = drift["x"] != 0 or drift["y"] != 0
    log(f"  differential drift ({drift['x']:.4f}, {drift['y']:.4f}) px/s, "
        f"reference = level {ref['level']} at t={t_ref:.1f}" if has_drift
        else "  no drift supplied — rigid Moon registration only")

    vx0 = vx1 = vy0 = vy1 = 0.0
    loaded = []
    W = H = None
    for sp in specs:
        img = read_xisf(sp["path"])
        if W is None:
            H, W = img.shape[:2]
        elif img.shape[:2] != (H, W):
            raise ValueError(f"geometry mismatch on {sp['path']}")
        ch = [img[:, :, c].astype(np.float32) for c in range(3)]

        dt = sp.get("t", 0.0) - t_ref
        # Prefer the shift measured by correlating against the reference:
        # independent per-level Moon fits disagree by hundreds of pixels at the
        # extremes of the ladder.
        base_x = sp["dx"] if sp.get("dx") is not None else (ref["cx"] - sp["cx"])
        base_y = sp["dy"] if sp.get("dy") is not None else (ref["cy"] - sp["cy"])
        dx = base_x - drift["x"] * dt
        dy = base_y - drift["y"] * dt
        if abs(dx) > 0.01 or abs(dy) > 0.01:
            ch = [translate_image(c, dx, dy) for c in ch]
        # Track where every level still has real data: a shifted level runs off
        # the sensor on one side, and that boundary is a step in the merge.
        vx0, vx1 = max(vx0, dx), min(vx1, dx)
        vy0, vy1 = max(vy0, dy), min(vy1, dy)

        limb_err = math.hypot(drift["x"], drift["y"]) * abs(dt)
        log(f"  L{sp['level']} {os.path.basename(sp['path'])} dt={dt:.1f}s "
            f"translateImage({dx:.2f}, {dy:.2f}) limb offset {limb_err:.1f} px")
        loaded.append({"level": sp["level"], "ch": ch, "limbErr": limb_err})

    # Average images sharing an exposure level; the group inherits the worst
    # limb offset among them.
    levels, i = [], 0
    while i < len(loaded):
        j = i
        while j < len(loaded) and loaded[j]["level"] == loaded[i]["level"]:
            j += 1
        if j - i == 1:
            levels.append(loaded[i])
        else:
            worst = max(loaded[k]["limbErr"] for k in range(i, j))
            acc = [sum(loaded[k]["ch"][c] for k in range(i, j)) / (j - i)
                   for c in range(3)]
            log(f"  averaged {j - i} images at level {loaded[i]['level']} "
                f"(worst limb offset {worst:.1f} px)")
            levels.append({"level": loaded[i]["level"], "ch": acc,
                           "limbErr": worst})
        i = j
    log(f"  {len(levels)} exposure levels")

    # Put every level on the shortest exposure's scale, per channel: the CFA
    # channels do not share a pedestal, so folding them into one number would
    # leave a radial colour cast behind.
    A = [[1.0] for _ in range(3)]
    B = [[0.0] for _ in range(3)]
    for k in range(1, len(levels)):
        for c in range(3):
            f = fit_affine(levels[k - 1]["ch"][c], levels[k]["ch"][c])
            if not f["a"] > 0:
                log(f"  WARNING: level {levels[k - 1]['level']} -> "
                    f"{levels[k]['level']} ch{c}: unusable overlap "
                    f"(n={f['n']}, span={f.get('span', 0):.2f}x) - assuming equal")
                f = {"a": 1.0, "b": 0.0}
            elif c == 1:
                log(f"  fit ch1 used {f['n']} samples spanning {f['span']:.1f}x")
            # Compose level k -> level 0: v0 = (vk - B)/A.
            A[c].append(A[c][k - 1] * f["a"])
            B[c].append(f["b"] + f["a"] * B[c][k - 1])
            if c == 1:
                log(f"  level {levels[k]['level']} = {f['a']:.3f} x level "
                    f"{levels[k - 1]['level']} + {f['b']:.3e}  "
                    f"(cumulative A={A[c][k]:.2f} B={B[c][k]:.3e})")

    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.hypot(xx - ref["cx"], yy - ref["cy"])
    out = [np.zeros((H, W), np.float64) for _ in range(3)]
    wsum = np.zeros((H, W), np.float64)

    for k, lv in enumerate(levels):
        g = lv["ch"][1]
        e = lv.get("limbErr", 0.0)
        gated = e > LIMB_TOL_PX
        if gated:
            g_lo = ref["r"] + e
            g_hi = g_lo + LIMB_FEATHER_PX
            log(f"  level {lv['level']} feathered in over r={g_lo:.0f}.."
                f"{g_hi:.0f} px (limb offset {e:.1f} px)")
            gate = ramp(dist, g_lo, g_hi)
        else:
            gate = 1.0
        w = gate * ramp(g, SIG_LO, SIG_HI) * (1 - ramp(g, SAT_LO, SAT_HI))
        wsum += w
        for c in range(3):
            out[c] += w * (lv["ch"][c] - B[c][k]) / A[c][k]

    # Pixels no level could represent — saturated even at the shortest exposure,
    # i.e. the chromosphere and any beads — fall back to the shortest exposure.
    orphan = wsum <= 0
    n_orphan = int(orphan.sum())
    for c in range(3):
        np.divide(out[c], wsum, out=out[c], where=~orphan)
        out[c][orphan] = levels[0]["ch"][c][orphan]
    log(f"  merged; {n_orphan} px ({100.0 * n_orphan / (W * H):.3f}%) fell back "
        f"to the shortest exposure")

    cx0, cy0 = int(math.ceil(max(0.0, vx0))), int(math.ceil(max(0.0, vy0)))
    cx1 = int(math.floor(W + min(0.0, vx1)))
    cy1 = int(math.floor(H + min(0.0, vy1)))
    cw, chh = cx1 - cx0, cy1 - cy0
    log(f"  common coverage {cw}x{chh} at ({cx0}, {cy0}), "
        f"trimmed {W - cw}x{H - chh} px")

    crop = [c[cy0:cy0 + chh, cx0:cx0 + cw].astype(np.float32) for c in out]
    write_xisf(cfg["out"], stack_channels(*crop))
    log(f"  saved {cfg['out']}")

    flat = flatten(crop, ref["cx"] - cx0, ref["cy"] - cy0, ref["r"], log=log)
    write_xisf(cfg["outFlat"], stack_channels(*flat))

    # Record the Moon in the TRIMMED frame's own coordinates: later stages
    # measure annuli about it, and the combine sidecars refer to the untrimmed
    # frame.
    side = os.path.splitext(cfg["outFlat"])[0] + "_moon.json"
    with open(side, "w", encoding="utf-8") as f:
        json.dump({"image": cfg["outFlat"], "width": cw, "height": chh,
                   "cx": ref["cx"] - cx0, "cy": ref["cy"] - cy0,
                   "radius": ref["r"]}, f)
    log(f"  saved {cfg['outFlat']}")
    log(f"  done in {time.time() - t0:.0f} s")
    return {"width": cw, "height": chh, "orphan": n_orphan}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="corona_hdr.json")
    args = ap.parse_args(argv)
    with open(args.config, encoding="utf-8-sig") as f:
        cfg = json.load(f)
    for k in ("out", "outFlat"):
        os.makedirs(os.path.dirname(os.path.abspath(cfg[k])), exist_ok=True)
    merge_hdr(cfg)


if __name__ == "__main__":
    main()
