"""Pass 1 of the timelapse: find the Sun in every frame — port of tl-centres.js.

Framing on the centroid of the lit region does not hold the Sun still. The
centroid of a crescent is not the centre of the disc it was cut from, it slides
further off as the Moon covers more of the Sun, and it jumps when the filter
comes off and the bright region becomes a corona. That is the bouncing. What is
actually stationary is the Sun's limb: a circle of known radius.

FILTERED FRAMES — limb points from an outward ray march (`sun_limb_points`),
then a centre-only fit at the known radius (`fixed_radius_fit`). Both of those
replaced earlier versions and the reasons are recorded at each function.

UNFILTERED FRAMES — no photosphere to fit, so the Moon's limb is found by
scoring candidate centres on the gradient of log brightness around a circle of
the Moon's radius. The two discs are nearly concentric during totality, so the
substitution costs at most (Rmoon - Rsun), about 13 px here.

Quality is reported per frame (inlier count, RMS residual, arc coverage in
degrees) so the smoothing pass can drop frames the fit could not constrain
rather than let them jerk the framing.

Note on lunation: `lunation.assemble.disk.analyze_disk` is the port of the same
`gif-frames.js` routine this script was written from, but it is not reusable
here. Its radius prior is a module constant (the lunar radius at TS-70 2x
drizzle scale, ~1058 px) with the fit clamped to +/-10% of it, and its limb
points come from the first/last row-scan crossing — which is precisely the
Sun-limb-plus-Moon-limb contamination the ray march below exists to fix.
"""

import argparse
import json
import math
import os
import time

import numpy as np

from . import serio
from .source import open_source
from . import paths

# Row scan step, in half-resolution pixels.
SCAN_STEP = 2

# Limb threshold as a fraction of the frame's bright level. Low on purpose: a
# soft limb ramps over several pixels and a high crossing sits inside the true
# limb, biasing the radius small.
EDGE_FRACTION = 0.10

# Kasa iterations and inlier tolerance schedule, in px.
FIT_ITERS = 12
TOL_EARLY = 40
TOL_LATE = 14

# Ring search (totality) parameters.
RING_SAMPLES = 360
COARSE_STEP = 8
REFINE_STEPS = [4, 2, 1]
TRACK_WINDOW = 70

# Sun/Moon apparent radius ratio on 2024-04-08: 1919" against 2010".
SUN_OVER_MOON = 0.9547

# Radius scan bracket for the bootstrap ring search, in PLANE px. These are the
# values for the camera this was written on (r=279); `tune()` replaces them with
# the same fractions of whatever radius the survey measured. Left as literals
# only so the module still runs standalone without a config.
R_MIN = 180
R_MAX = 340
R_BOOT_MIN = 60.0
R_PRIOR = 279.0

# A fit this poorly constrained is not trusted; the frame is re-acquired with a
# global ring search before refitting.
REACQUIRE_ARC = 100
REACQUIRE_N = 60

__all__ = ["find_centres", "log_gradient", "sun_limb_points", "fixed_radius_fit",
           "kasa_fit", "ring_search", "seed_disc"]


def _jsround(a):
    """JavaScript Math.round: half away from zero toward +inf, not banker's.

    np.rint rounds halves to even, which disagrees with the PJSR original on
    exact .5 offsets. Those do occur — the ring and ray offsets below are
    products of a cosine and an integer radius.
    """
    return np.floor(np.asarray(a, dtype=np.float64) + 0.5).astype(np.int64)


# ------------------------------------------------------------------ gradient

def tune(out_dir, log=None):
    """Resolve the acquisition bracket against the surveyed disc radius.

    Everything else in this module is already scale-free - it fits a circle to
    limb points and holds the radius. The exception was the ONE global search
    that bootstraps that radius, which scanned a fixed 180-340 px and clamped
    the bounding-box guess at 60 px. On a disc smaller than 180 px it could only
    return a wrong answer, and it did: a 48 px disc came back as 180 px, which
    then poisoned every panel that stands on the Sun's limb.
    """
    global R_MIN, R_MAX, R_BOOT_MIN, R_PRIOR
    from .params import load

    P = load(out_dir, create=False)
    R_PRIOR = P.radius_px
    R_MIN = max(8, int(P.px("centres.r_search_min_r")))
    R_MAX = max(R_MIN + 8, int(P.px("centres.r_search_max_r")))
    R_BOOT_MIN = max(4.0, P.px("centres.r_bootstrap_min_r"))
    if log:
        log(f"  tuned to r={R_PRIOR:.0f}px: radius search {R_MIN}-{R_MAX} px")
    return P


def log_gradient(g):
    """Gradient magnitude of log brightness, softened near zero.

    A plain log blows up in empty sky: the values there are noise about zero, and
    the log of a ratio of two small noisy numbers is large, so blank sky scores
    higher than the real limb. That is not hypothetical — it pinned the totality
    search against the right edge of the frame, reporting the Moon at x=1918 of
    1920 when it was actually at x=755, and made 162 totality frames look like
    they had a Sun clipped by the sensor.

    Flooring at the frame's own median before the log keeps the response
    logarithmic across the corona, where the dynamic range needs it, while
    damping anything at or below sky level.
    """
    flat = g.reshape(-1)
    # Cheap median: a strided sample is plenty for a floor. Indexed rather than
    # np.median so an even-length sample matches the original's sort-and-halve.
    samp = np.sort(flat[::37])
    floor = max(float(samp[samp.size // 2]), 1e-7)

    L = np.log(g.astype(np.float32) + np.float32(floor))
    out = np.zeros_like(L)
    dx = L[1:-1, 2:] - L[1:-1, :-2]
    dy = L[2:, 1:-1] - L[:-2, 1:-1]
    out[1:-1, 1:-1] = np.hypot(dx, dy)
    return out


# --------------------------------------------------------------- limb points

def sun_limb_points(g, cx, cy, R, nb=360, d=2, tiny=1e-6):
    """Sun-limb points only: along each ray outward, the OUTERMOST falling edge.

    This replaces a row scan that took the first and last bright pixel on each
    line. That fed the circle fit both limbs at once — the Sun's where it is
    uncovered and the Moon's where it is not — and the radius prior does not sort
    them out, because on a thin crescent the two arcs are only a few pixels apart
    and both sit inside the inlier tolerance. The resulting bias depended on the
    crescent's geometry, so it changed from capture to capture: the Sun sat up to
    76 px off centre and the offset jumped every time a new capture began.

    Marching outward fixes it by construction. Past the Sun's limb there is only
    sky, so the last strong falling edge on any ray is the photosphere's edge and
    nothing else. Rays crossing only the Moon-covered part find no edge and
    contribute nothing.
    """
    h, w = g.shape
    r_lo = max(4, int(math.floor(0.55 * R)))
    r_hi = int(math.ceil(1.45 * R))
    if r_hi < r_lo:
        return np.zeros((0, 2))

    rs = np.arange(r_lo, r_hi + 1)
    ang = 2.0 * np.pi * np.arange(nb) / nb
    ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]

    xi, yi = _jsround(cx + ca * (rs - d)), _jsround(cy + sa * (rs - d))
    xo, yo = _jsround(cx + ca * (rs + d)), _jsround(cy + sa * (rs + d))

    bad_i = (xi < 0) | (yi < 0) | (xi >= w) | (yi >= h)
    bad_o = (xo < 0) | (yo < 0) | (xo >= w) | (yo >= h)

    # An out-of-frame INNER sample skips that radius; an out-of-frame OUTER
    # sample ends the ray, so everything beyond the first one is dropped.
    stop = np.where(bad_o.any(axis=1), bad_o.argmax(axis=1), rs.size)
    valid = (~bad_i) & (np.arange(rs.size)[None, :] < stop[:, None])

    gi = g[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)]
    go = g[np.clip(yo, 0, h - 1), np.clip(xo, 0, w - 1)]
    drop = np.where(valid, np.log(gi + tiny) - np.log(go + tiny), -np.inf)

    best = drop.max(axis=1)
    # The original tests `drop >= best`, so ties take the outermost radius;
    # find the LAST index attaining the maximum, not the first.
    last = rs.size - 1 - (drop[:, ::-1] == best[:, None]).argmax(axis=1)

    # A real limb is a large multiplicative drop; glow gradients are gentle.
    keep = best > 0.9
    if not keep.any():
        return np.zeros((0, 2))
    br = rs[last[keep]]
    return np.stack([cx + ca[keep, 0] * br, cy + sa[keep, 0] * br], axis=1)


def limb_points(g):
    """First and last bright pixel on each row.

    Used only to bootstrap the radius on the least eclipsed frame, where the disc
    is nearly full and both boundaries are essentially the same circle.
    """
    h, w = g.shape
    thr = float(g.max()) * EDGE_FRACTION
    sub = g[::SCAN_STEP, ::SCAN_STEP]
    hit = sub > thr
    rows = np.nonzero(hit.any(axis=1))[0]
    if rows.size == 0:
        return np.zeros((0, 2)), w / 2.0, h / 2.0, 0.0

    first = hit.argmax(axis=1)[rows] * SCAN_STEP
    last = (hit.shape[1] - 1 - hit[:, ::-1].argmax(axis=1))[rows] * SCAN_STEP
    ys = rows * SCAN_STEP
    pts = np.concatenate([np.stack([first, ys], axis=1),
                          np.stack([last, ys], axis=1)]).astype(np.float64)

    bx0, bx1 = float(first.min()), float(last.max())
    by0, by1 = float(ys.min()), float(ys.max())
    return (pts, (bx0 + bx1) / 2, (by0 + by1) / 2,
            max(bx1 - bx0, by1 - by0) / 2)


# ---------------------------------------------------------------------- fits

def _quality(pts, cx, cy, r, tol=8.0):
    """Inlier count, RMS residual and arc coverage in degrees.

    A thin crescent constrains very little and the caller needs to know that
    rather than be handed a confident wrong centre.
    """
    if len(pts) == 0:
        return 0, 0.0, 0
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    d = np.hypot(dx, dy) - r
    inl = np.abs(d) < tol
    n = int(inl.sum())
    if n == 0:
        return 0, 0.0, 0
    rms = float(np.sqrt((d[inl] ** 2).mean()))
    a = np.arctan2(dy[inl], dx[inl])
    bins = np.zeros(36, dtype=bool)
    bins[(np.floor((a + np.pi) / (2 * np.pi) * 36).astype(int) % 36)] = True
    return n, rms, int(bins.sum()) * 10


def fixed_radius_fit(pts, cx0, cy0, R):
    """Centre-only fit at a KNOWN radius.

    A three-parameter circle fit needs a decent spread of arc to pin the centre.
    As the crescent thins the usable arc shrinks to well under half the limb and
    the fit becomes ill conditioned along the direction perpendicular to that arc
    — radius and centre trade off, so the centre slides. Because the bias depends
    on the crescent's orientation it landed differently in every capture, which
    is exactly the 10-30 px step seen at capture boundaries.

    The Sun's angular radius does not change over 45 minutes. Fixing it removes
    the degenerate direction entirely: each limb point says "the centre lies R
    inward along my own ray", and the answer is their mean. Two unknowns instead
    of three, and no trade-off left to go wrong.
    """
    cx, cy = float(cx0), float(cy0)
    for it in range(8):
        tol = TOL_EARLY if it < 3 else TOL_LATE
        dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
        d = np.hypot(dx, dy)
        m = (d >= 1) & (np.abs(d - R) <= tol)
        if int(m.sum()) < 12:
            break
        # Pull each point back along its own ray by exactly R.
        cx = float(np.mean(pts[m, 0] - R * dx[m] / d[m]))
        cy = float(np.mean(pts[m, 1] - R * dy[m] / d[m]))

    n, rms, arc = _quality(pts, cx, cy, R)
    return {"cx": cx, "cy": cy, "r": float(R), "n": n, "rms": rms, "arc": arc}


def kasa_fit(pts, cx0, cy0, r0, r_lo, r_hi):
    """Kasa algebraic circle fit, iterated with a shrinking inlier tolerance and
    the radius clamped to a prior."""
    cx, cy, r = float(cx0), float(cy0), float(r0)
    for it in range(FIT_ITERS):
        tol = TOL_EARLY if it < 3 else TOL_LATE
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        sel = pts[np.abs(d - r) <= tol]
        if len(sel) < 20:
            break
        x, y = sel[:, 0], sel[:, 1]
        z = x * x + y * y
        A = np.column_stack([x, y, np.ones(len(sel))])
        try:
            sol, *_ = np.linalg.lstsq(A, z, rcond=None)
        except np.linalg.LinAlgError:
            break
        cx, cy = sol[0] / 2, sol[1] / 2
        r = math.sqrt(max(1.0, sol[2] + cx * cx + cy * cy))
        r = max(r_lo, min(r_hi, r))

    n, rms, arc = _quality(pts, cx, cy, r)
    return {"cx": cx, "cy": cy, "r": r, "n": n, "rms": rms, "arc": arc}


# -------------------------------------------------------------- ring search

def seed_disc(g, R, ds=8, ang=32):
    """Locate the dark disc sitting inside a bright ring, on a decimated copy.

    A bare gradient ring search is not reliable during totality: the corona
    covers most of the frame and there are plenty of circular-ish edges in it,
    and the search wandered by more than a thousand pixels between adjacent
    frames. Matching the actual structure — dark inside, bright around — has no
    such ambiguity. Radius is known, so only the centre is searched.
    """
    h, w = g.shape
    sh, sw = h // ds, w // ds
    small = g[:sh * ds, :sw * ds].reshape(sh, ds, sw, ds).mean(axis=(1, 3))

    rs = R / ds
    r_in, r_out = rs * 0.70, rs * 1.15
    a = 2.0 * np.pi * np.arange(ang) / ang
    # Candidate centres are integers, so the sample offsets are the same for all
    # of them and can be rounded once.
    dxo, dyo = _jsround(np.cos(a) * r_out), _jsround(np.sin(a) * r_out)
    dxi, dyi = _jsround(np.cos(a) * r_in), _jsround(np.sin(a) * r_in)

    m = int(math.ceil(r_out)) + 1
    if sh - m <= m or sw - m <= m:
        return {"cx": w / 2.0, "cy": h / 2.0, "score": -np.inf}
    cys, cxs = np.arange(m, sh - m), np.arange(m, sw - m)
    YY, XX = np.meshgrid(cys, cxs, indexing="ij")

    ring = np.zeros(YY.shape, dtype=np.float64)
    disc = np.zeros(YY.shape, dtype=np.float64)
    for k in range(ang):
        ring += small[YY + dyo[k], XX + dxo[k]]
        disc += small[YY + dyi[k], XX + dxi[k]]
    score = (ring - disc) / ang

    # Row-major argmax matches the original's cy-outer/cx-inner scan with a
    # strict >, which keeps the first of any tie.
    j = int(np.argmax(score))
    iy, ix = divmod(j, score.shape[1])
    return {"cx": float((cxs[ix] + 0.5) * ds),
            "cy": float((cys[iy] + 0.5) * ds),
            "score": float(score[iy, ix])}


def _ring_scores(grad, cxs, cys, dxk, dyk, min_n):
    """Mean gradient around a ring, for many integer candidate centres at once.

    Candidates below `min_n` in-frame samples score -inf. The Moon's limb is a
    closed circle inside the frame during totality, so almost the whole ring is
    demanded; accepting half a ring is what let solutions hanging off the frame
    edge compete at all.
    """
    h, w = grad.shape
    X = cxs[:, None] + dxk[None, :]
    Y = cys[:, None] + dyk[None, :]
    ok = (X >= 1) & (Y >= 1) & (X < w - 1) & (Y < h - 1)
    v = np.where(ok, grad[np.clip(Y, 0, h - 1), np.clip(X, 0, w - 1)], 0.0)
    n = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = v.sum(axis=1) / np.maximum(n, 1)
    return np.where(n >= min_n, s, -np.inf)


def ring_search(grad, R, seed=None, win=None, chunk=8192):
    """Coarse-to-fine search for the centre of a ring of known radius R."""
    h, w = grad.shape
    a = 2.0 * np.pi * np.arange(RING_SAMPLES) / RING_SAMPLES
    dxk, dyk = _jsround(np.cos(a) * R), _jsround(np.sin(a) * R)
    min_n = RING_SAMPLES * 0.95

    x0, x1, y0, y1 = 0, w, 0, h
    if seed is not None and win is not None:
        x0 = max(0, int(round(seed["cx"] - win)))
        x1 = min(w, int(round(seed["cx"] + win)))
        y0 = max(0, int(round(seed["cy"] - win)))
        y1 = min(h, int(round(seed["cy"] + win)))

    gx = np.arange(x0, x1, COARSE_STEP)
    gy = np.arange(y0, y1, COARSE_STEP)
    bx, by = (x0 + x1) >> 1, (y0 + y1) >> 1
    bs = -np.inf
    if gx.size and gy.size:
        YY, XX = np.meshgrid(gy, gx, indexing="ij")
        fx, fy = XX.reshape(-1), YY.reshape(-1)
        best_j, best_s = -1, -np.inf
        for o in range(0, fx.size, chunk):
            sc = _ring_scores(grad, fx[o:o + chunk], fy[o:o + chunk],
                              dxk, dyk, min_n)
            j = int(np.argmax(sc))
            if sc[j] > best_s:
                best_s, best_j = float(sc[j]), o + j
        if best_j >= 0:
            bx, by, bs = int(fx[best_j]), int(fy[best_j]), best_s

    # The refine walk is deliberately sequential: the original updates the centre
    # inside the loop, so later probes are taken around the new best, not the old.
    for i, step in enumerate(REFINE_STEPS):
        span = COARSE_STEP if i == 0 else REFINE_STEPS[i - 1]
        for dy in range(-span, span + 1, step):
            for dx in range(-span, span + 1, step):
                sc = float(_ring_scores(grad, np.array([bx + dx]),
                                        np.array([by + dy]), dxk, dyk, min_n)[0])
                if sc > bs:
                    bs, bx, by = sc, bx + dx, by + dy

    # Parabolic sub-pixel refinement against the four neighbours.
    probe = _ring_scores(grad,
                         np.array([bx - 1, bx + 1, bx, bx]),
                         np.array([by, by, by - 1, by + 1]),
                         dxk, dyk, min_n)
    xm, xp, ym, yp = (float(v) for v in probe)
    ddx, ddy = xm + xp - 2 * bs, ym + yp - 2 * bs
    ox = 0.5 * (xm - xp) / ddx if ddx < 0 else 0.0
    oy = 0.5 * (ym - yp) / ddy if ddy < 0 else 0.0
    if not abs(ox) <= 1:
        ox = 0.0
    if not abs(oy) <= 1:
        oy = 0.0
    return {"cx": bx + ox, "cy": by + oy, "score": bs}


# ------------------------------------------------------------------ the pass

def find_centres(frames, log=print):
    """Fit the Sun (or, during totality, the Moon) in every frame.

    `frames` is the `frames` array of timelapse.json: each entry needs `src`,
    `file`, `index` and `state`. Returns the centres.json payload.
    """
    t0 = time.time()
    log(f"centres: {len(frames)} frames")

    ser = None
    cur_path = ""
    last_file = ""
    last = None
    r_sun = 0.0
    r_moon = 0
    w = h = 0
    results = []

    try:
        for k, fr in enumerate(frames):
            if fr["src"] != cur_path:
                if ser is not None:
                    ser.close()
                ser = open_source(fr["src"])
                # PLANE size, which is half the sensor on a CFA camera and the
                # whole of it on an already-demosaiced frame. This used to shift
                # right unconditionally, so every image-sequence run measured
                # its geometry in a box half the size of the frames it was
                # actually reading.
                w = int(round(ser.raw_width * ser.plane_scale))
                h = int(round(ser.raw_height * ser.plane_scale))
                cur_path = fr["src"]
                # The mount was nudged between captures, so a track must not be
                # carried across a file boundary.
                if fr["file"] != last_file:
                    last, last_file = None, fr["file"]
                log(f"  open {fr['file']}")

            g = ser.green(fr["index"])

            if fr["state"] == "unfiltered":
                if r_moon <= 0:
                    r_moon = int(round((r_sun or R_PRIOR) / SUN_OVER_MOON))
                    log(f"  moon limb radius {r_moon} px (plane)")
                grad = log_gradient(g)
                seed = seed_disc(g, r_moon)
                c = ring_search(grad, r_moon, seed, TRACK_WINDOW)
                last = c
                res = {"cx": c["cx"], "cy": c["cy"], "r": float(r_moon),
                       "n": 0, "rms": 0.0, "arc": 360, "method": "ring"}
            else:
                pts, bcx, bcy, bbox_r = limb_points(g)
                if r_sun <= 0:
                    # Bootstrap on the first filtered frame, the least eclipsed
                    # one available: a wide clamp seeded from the bounding box.
                    # That seed is only good while the disc is nearly full — it
                    # sits ON the crescent later — which is exactly why the
                    # radius is measured once, here, and then held.
                    wide = kasa_fit(pts, bcx, bcy, max(bbox_r, R_BOOT_MIN),
                                    R_BOOT_MIN, min(w, h) / 2)
                    if wide["arc"] < 120:
                        grad = log_gradient(g)
                        best = None
                        for R in range(R_MIN, R_MAX + 1, 4):
                            c = ring_search(grad, R)
                            if best is None or c["score"] > best["score"]:
                                best = {**c, "r": R}
                        wide = kasa_fit(pts, best["cx"], best["cy"], best["r"],
                                        0.85 * best["r"], 1.15 * best["r"])
                    r_sun = wide["r"]
                    last = {"cx": wide["cx"], "cy": wide["cy"], "score": 1.0}
                    log(f"  sun limb radius measured at {r_sun:.1f} px (plane)"
                        f" = {r_sun / ser.plane_scale:.0f} px on the sensor"
                        f"  [arc {wide['arc']} deg, rms {wide['rms']:.2f}]")

                seed_x = last["cx"] if last else bcx
                seed_y = last["cy"] if last else bcy
                # Two passes: the ray march needs a seed inside the disc, and its
                # own answer is a better seed than the one it started from.
                f = None
                for _ in range(2):
                    lp = sun_limb_points(g, seed_x, seed_y, r_sun)
                    if len(lp) < 20:
                        break
                    f = fixed_radius_fit(lp, seed_x, seed_y, r_sun)
                    seed_x, seed_y = f["cx"], f["cy"]
                if f is None:
                    f = kasa_fit(pts, seed_x, seed_y, r_sun,
                                 0.94 * r_sun, 1.06 * r_sun)
                # A thin crescent constrains almost nothing from a stale seed.
                # Rather than emit a confident wrong centre, re-acquire globally.
                if f["arc"] < REACQUIRE_ARC or f["n"] < REACQUIRE_N:
                    grad = log_gradient(g)
                    c = ring_search(grad, r_sun)
                    lp = sun_limb_points(g, c["cx"], c["cy"], r_sun)
                    if len(lp) >= 20:
                        f2 = fixed_radius_fit(lp, c["cx"], c["cy"], r_sun)
                        if f2["arc"] > f["arc"] or f2["n"] > f["n"]:
                            f = f2
                last = {"cx": f["cx"], "cy": f["cy"], "score": 1.0}
                res = {**f, "method": "kasa"}

            results.append({"i": k, "file": fr["file"], "index": fr["index"],
                            "state": fr["state"], "cx": res["cx"], "cy": res["cy"],
                            "r": res["r"], "n": res["n"], "rms": res["rms"],
                            "arc": res["arc"], "method": res["method"]})

            if (k + 1) % 200 == 0:
                log(f"  {k + 1}/{len(frames)} "
                    f"({(time.time() - t0) / (k + 1):.2f} s/frame)")
    finally:
        if ser is not None:
            ser.close()

    log(f"  done in {(time.time() - t0) / 60:.1f} min")
    return {"width": w, "height": h, "rSun": r_sun, "rMoon": r_moon,
            "centres": results}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=paths.in_out("configs", "timelapse.json"))
    ap.add_argument("--out", default=paths.in_out("diag", "centres.json"))
    ap.add_argument("--limit", type=int, default=None,
                    help="fit only the first N frames (smoke tests)")
    ap.add_argument("--data-dir", default=None,
                    help="read the captures from here instead of the paths "
                         "baked into the config")
    args = ap.parse_args(argv)

    # The output path is <out>/diag/centres.json and the config is
    # <out>/configs/timelapse.json; either gives the output root the tuning
    # lives in. Take it from the config, which every caller passes.
    tune(os.path.dirname(os.path.dirname(os.path.abspath(args.config))),
         log=print)

    with open(args.config, encoding="utf-8-sig") as f:
        cfg = json.load(f)
    serio.restage(cfg, args.data_dir)
    frames = cfg["frames"][:args.limit] if args.limit else cfg["frames"]

    out = find_centres(frames)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"wrote {args.out}  rSun={out['rSun']:.3f} rMoon={out['rMoon']}")


if __name__ == "__main__":
    main()
