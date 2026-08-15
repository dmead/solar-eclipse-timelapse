"""Stage F pass 3 - place zoomed panels on the interesting things.

The main view is already 1:1 with the sensor: the video is a 2240x1680 crop of the
3840x2160 fine grid, so one video pixel is one sensor pixel. That means an inset
only earns its place by magnifying PAST 1:1, and the detail to magnify into has to
come from the drizzle - which is why every frame is stacked, not just totality.

Nothing here is a hand-placed box. Each panel follows a feature that is computed
per frame from geometry already measured, and carries the name of what it is
following:

  sunspot        found once by detection and then held in SUN coordinates - the
                 video is Sun-stabilised and solar rotation over 46 minutes is
                 about half a degree, so it simply stays put
  lunar limb     the Moon's deepest incursion into the disc, where lunar terrain
                 is silhouetted against the photosphere
  upper cusp     the two points where the Sun's and Moon's limbs cross, named by
  lower cusp     where they sit in the picture - see the note in main() for why
                 not by which one leads
  prominence     through totality, where there is no photosphere and no cusps

The count is NOT fixed at four. A feature is emitted only while its subject
exists: the sunspot goes behind the Moon and comes back out, the cusps exist only
while the limbs actually intersect, and a totality exposure level may show fewer
than four prominences worth pointing at. Panels are assigned to whichever corners
are nearest, so an emitted feature can land in any of the four.

The cusps need the Moon's centre - which no stage measures during the partial
phases, since the detector is fitting the photosphere there. It is recovered here
by fitting the terminator the Moon casts on the disc, which needs no ephemeris.

    D:/projects/umbra/venv/Scripts/python.exe gen_insets.py --out S:/solar-eclipse/out

Run after smooth_track.py; rewrites configs/timelapse.json with an "insets" list
on every frame.
"""

import argparse
import datetime
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serlib import SerFile  # noqa: E402

# Panel geometry, in FINE (sensor) pixels. The source box is the panel divided by
# the zoom, so a bigger zoom means a tighter box.
PANEL_PX = 420
ZOOM = 3.0

# Cusps get their own, tighter box. A horn is a needle: at the zoom that suits a
# sunspot group or a prominence it is a thin bright sliver crossing one corner of
# the panel, and most of the box is empty sky.
CUSP_ZOOM = 5.0

# Sunspot search: how far out to look, and the background scale it is measured
# against. The spot is a local darkening of a few px on a disc whose brightness
# falls off toward the limb, so a plain minimum finds the limb instead - the
# background has to be divided out first.
SPOT_MAX_R = 0.90
MIN_RING_PX = 20
# Frames sampled across the partial phases when hunting for the sunspot.
SPOT_TRY_FRAMES = 9

# Prominence search: an annulus about the Moon's centre, in units of the MOON's
# radius. Picked maxima must be this far apart so four panels do not all land on
# one prominence.
#
# The inner bound has to reach well inside the Moon's radius, and 0.97 did not.
# Prominences stand on the SUN's limb, and the Sun's disc is the smaller of the
# two - 1919" against 2010", a ratio of 0.955 - so a prominence rooted at the
# photosphere already measures inside 0.96 R before anything goes wrong. On top of
# that the Moon's centre here is a fitted track carrying about 7 px of residual and
# r_moon is itself an estimate. Measured on 14_14_36 f340, the third strongest
# feature in the frame - an obvious prominence at 12 o'clock, plainly visible in the
# render - sits at 0.939 R and was thrown away by the inner bound while panels went
# to peaks a quarter as strong. Widening costs nothing: behind the Moon there is no
# Halpha, so the lunar disc cannot win an argmax on R - k*G.
PROM_R_INNER = 0.90
PROM_R_OUTER = 1.25
PROM_MIN_SEP_PX = 90.0

# How far a peak must stand above the annulus, in robust sigmas, to be called a
# prominence at all. Without this the search always returns four answers, so on a
# level showing two prominences the other two panels magnified inner corona and
# put a box around nothing.
PROM_MIN_SNR = 8.0

# Terminator fits worse than this are not the Moon's limb and are discarded.
MOON_FIT_MAX_RMS = 12.0
MOON_TRY_FRAMES = 14


def box_blur(a, k):
    """Separable box filter via summed-area, so no scipy dependency."""
    h = k // 2
    p = np.pad(a, h + 1, mode="edge")
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    y, x = a.shape
    ys, xs = np.arange(y) + k, np.arange(x) + k
    y0, x0 = ys - k, xs - k
    s = (c[np.ix_(ys, xs)] - c[np.ix_(y0, xs)]
         - c[np.ix_(ys, x0)] + c[np.ix_(y0, x0)])
    return s / (k * k)


def green_plane(ser, index):
    """Superpixel G at half resolution, matching the detector's grid."""
    raw = np.frombuffer(ser.read_frame(index), dtype=np.uint16)
    a = raw.reshape(ser.height, ser.width).astype(np.float32)
    return 0.5 * (a[0::2, 1::2] + a[1::2, 0::2])


def halpha_plane(ser, index):
    """R with the white corona subtracted, so only Halpha is left standing.

    Prominences are chromospheric Halpha at 656 nm and land almost entirely in R;
    the corona is broadly white. Searching plain brightness therefore finds the
    brightest CORONA, which is what the first attempt did - all four panels landed
    on inner-corona peaks and not one on a prominence. Scaling G to match R over
    the limb ring and subtracting cancels anything white and leaves the emission.
    """
    raw = np.frombuffer(ser.read_frame(index), dtype=np.uint16)
    a = raw.reshape(ser.height, ser.width).astype(np.float32)
    r = a[0::2, 0::2]
    g = 0.5 * (a[0::2, 1::2] + a[1::2, 0::2])
    k = np.median(r) / max(np.median(g), 1e-3)
    return r - k * g


def find_sunspot(data_dir, frame, cx, cy, r_sun):
    """Darkest feature on the photosphere, as an offset from the Sun's centre."""
    with SerFile(os.path.join(data_dir, frame["file"])) as ser:
        g = green_plane(ser, frame["index"])
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - cx, yy - cy)

    """
    Find the photosphere by BRIGHTNESS, and normalise it by a RADIAL profile.

    Both halves were wrong on the first two attempts, in opposite ways.

    Masking the Moon by its modelled position is not good enough. The drift model
    is accurate to a few pixels, and a few pixels at the terminator is the
    difference between photosphere and shadow - and the shadow is far darker than
    any sunspot, so the search walks straight to the Moon's edge. Where the
    photosphere IS is not a geometric question here; it is simply the bright part.

    And a local box mean cannot normalise limb darkening, because a box mean
    removes a constant while limb darkening is a steep gradient. Dividing by one
    leaves a ratio that falls off toward the limb, so the darkest pixel is just
    the outermost - which is how a "sunspot" at 0.89 R with 0.07 of the local mean
    came back. Limb darkening is radially symmetric about the Sun's centre, so a
    mean in radius bins removes it exactly.
    """
    inside = rr < r_sun
    lit = inside & (g > 0.5*np.percentile(g[inside], 99))
    rb = rr.astype(np.int32)
    nb = int(r_sun) + 1
    cnt = np.bincount(rb[lit], minlength=nb)[:nb]
    tot = np.bincount(rb[lit], weights=g[lit], minlength=nb)[:nb]
    # A radius sampled by only a handful of lit pixels has a meaningless mean.
    prof = np.where(cnt > MIN_RING_PX, tot/np.maximum(cnt, 1), 0.0)
    bg = prof[np.clip(rb, 0, nb - 1)]
    rel = np.where((bg > 1e-3) & lit, g/np.maximum(bg, 1e-3), 1.0)
    # A single dark pixel is a defect, not a sunspot; require it to survive a blur.
    rel = box_blur(rel, 3)
    rel = np.where(rr < SPOT_MAX_R*r_sun, rel, 1.0)
    j = int(np.argmin(rel))
    sy, sx = j // w, j % w
    return float(sx - cx), float(sy - cy), float(rel[sy, sx])


def best_sunspot(data_dir, partials, r_sun, n_try):
    """Run the spot search over several frames and keep the clearest detection.

    Which frame is used matters. The two 30 s test runs at the start are exposed
    close to saturation, which flattens the contrast a sunspot depends on, and
    late in the partial phase the spot may already be behind the Moon. Rather than
    privilege any particular capture, try a spread of them and keep the deepest.
    """
    best = None
    step = max(1, len(partials)//n_try)
    for f in partials[::step]:
        try:
            ox, oy, depth = find_sunspot(data_dir, f, f["cx"], f["cy"], r_sun)
        except Exception:
            continue
        if best is None or depth < best[2]:
            best = (ox, oy, depth, f)
    return best


def find_prominences(data_dir, frame, mx, my, r_moon, n):
    """Offsets from the Moon's centre of up to n prominences, with their strength.

    Searched in TWO dimensions over an annulus, not along a single ring. A
    prominence is an arch standing off the limb, so its brightest point is not at
    any one radius - sampling a ring at 1.03 R put the panel just below the
    prominence it was supposed to be centred on. Taking the maximum over the whole
    annulus lands on the feature itself.

    Peaks are cut off at PROM_MIN_SNR rather than always returning n of them. An
    argmax always has an answer; whether that answer is a prominence is a separate
    question, and the annulus's own median and MAD answer it - a prominence stands
    tens of sigmas over the corona behind it, and a corona ripple does not.
    """
    with SerFile(os.path.join(data_dir, frame["file"])) as ser:
        g = halpha_plane(ser, frame["index"])
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - mx, yy - my)
    band = (rr > PROM_R_INNER*r_moon) & (rr < PROM_R_OUTER*r_moon)
    v = np.where(band, g, -np.inf)

    ref = g[band]
    med = float(np.median(ref))
    sigma = 1.4826*float(np.median(np.abs(ref - med))) or 1.0

    out = []
    for _ in range(n):
        j = int(np.argmax(v))
        py, px = j // w, j % w
        if not np.isfinite(v[py, px]):
            break
        snr = (float(v[py, px]) - med)/sigma
        if snr < PROM_MIN_SNR:
            break
        out.append((float(px - mx), float(py - my), snr))
        # Blank a neighbourhood so the next pick is a different prominence and
        # not the pixel next door.
        v[np.hypot(xx - px, yy - py) < PROM_MIN_SEP_PX] = -np.inf
    return out


def fit_moon(data_dir, frame, cx, cy, r_sun, r_moon):
    """Moon centre for one partial frame, from the terminator it casts.

    The boundary between lit and unlit INSIDE the Sun's disc is the Moon's limb,
    so its centre is recoverable directly from the picture - no ephemeris and no
    drift model. Points on a circle of known radius satisfy
    2 P.M - (|M|^2 - R^2) = |P|^2, which is linear in (Mx, My, k), so one
    least-squares solve gives the centre.
    """
    with SerFile(os.path.join(data_dir, frame["file"])) as ser:
        g = green_plane(ser, frame["index"])
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - cx, yy - cy)
    inside = rr < r_sun
    lit = inside & (g > 0.5*np.percentile(g[inside], 99))

    # Terminator: unlit pixels inside the disc with a lit neighbour. Kept well
    # clear of the solar limb, or the limb itself joins the fit and drags it.
    core = inside & (rr < 0.97*r_sun)
    dark = core & ~lit
    nb = np.zeros_like(lit)
    nb[1:, :] |= lit[:-1, :]; nb[:-1, :] |= lit[1:, :]
    nb[:, 1:] |= lit[:, :-1]; nb[:, :-1] |= lit[:, 1:]
    ys, xs = np.nonzero(dark & nb)
    if len(xs) < 50:
        return None

    A = np.column_stack([2*xs, 2*ys, -np.ones(len(xs))]).astype(np.float64)
    b = (xs.astype(np.float64)**2 + ys.astype(np.float64)**2)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    mx, my = float(sol[0]), float(sol[1])
    rms = float(np.sqrt(np.mean((np.hypot(xs - mx, ys - my) - r_moon)**2)))
    return mx, my, rms, len(xs)


def fit_moon_track(data_dir, partials, r_sun, r_moon, n_try):
    """Linear Moon-vs-Sun motion, fitted to measurements on sampled frames.

    Measuring every frame would mean reading all of them again. The relative
    motion is very nearly linear over 46 minutes, so a spread of samples and a
    straight line through them is both cheaper and steadier than per-frame fits,
    and the residual reports whether that assumption held.
    """
    step = max(1, len(partials)//n_try)
    ts, ox, oy = [], [], []
    for f in partials[::step]:
        r = fit_moon(data_dir, f, f["cx"], f["cy"], r_sun, r_moon)
        if r is None or r[2] > MOON_FIT_MAX_RMS:
            continue
        ts.append(f["utc"])
        ox.append(r[0] - f["cx"])
        oy.append(r[1] - f["cy"])
    if len(ts) < 3:
        raise SystemExit("could not measure the Moon on enough frames")
    t = np.array(ts); vx = np.polyfit(t, np.array(ox), 1); vy = np.polyfit(t, np.array(oy), 1)
    rx = np.array(ox) - np.polyval(vx, t)
    ry = np.array(oy) - np.polyval(vy, t)
    resid = float(np.sqrt(np.mean(rx**2 + ry**2)))
    return vx, vy, len(ts), resid


# Cusp search along the Sun's limb: how many samples around the circle, how far
# inside the limb to sample, and the smallest lit arc worth calling a crescent.
CUSP_SAMPLES = 1440
CUSP_MIN_ARC = 6

# Radial band scanned at each angle, in units of r_sun, and how finely.
#
# NOT one circle. One circle works while the crescent is fat and fails exactly
# where it matters: near second contact the lit band is a few px wide radially and
# does not sit at a fixed fraction of r_sun, so a single circle clips it obliquely
# and the brightness falls before the horn ends - every box then lands SHORT of the
# tip. This was not a small bias, it was the dominant NOISE source too. Scanning a
# band and keeping the brightest sample per angle took the per-capture scatter from
# 2-5 px to 0.2-1.0 px, and on the thinnest crescent from 8.0 px to 1.9 px.
CUSP_R_LO = 0.93
CUSP_R_HI = 1.01
CUSP_R_STEPS = 17

# Fraction of the limb's own brightness that still counts as lit. Low on purpose,
# for the same reason tl-centres.js takes its limb points low: the horn TAPERS, so
# the falloff is gradual and a half-height crossing sits inside the visible tip.
CUSP_EDGE_FRAC = 0.08

# Fewest measured frames a capture needs before its cusps are worth fitting, and
# the count above which a quadratic is used instead of a line.
CUSP_MIN_FRAMES = 10
CUSP_QUAD_MIN = 25

# How far to push the cusp box back along the limb, INTO the crescent, as a
# fraction of the box half-width. Centred exactly on the tip, half the panel is
# empty sky beyond the horn; sliding it back fills the box with the horn while
# keeping the tip comfortably inside.
CUSP_INSET_FRAC = 0.35

# Fraction of the scanned band's depth that counts as "the horn has opened up",
# and how many of those opening-lengths the box should span. Bounds on the
# resulting half-box, in superpixel px, so neither a hair nor a blunt corner can
# drive the panel to a silly magnification.
CUSP_THICK_FRAC = 0.5
CUSP_BOX_K = 1.3
CUSP_HALF_MIN = 14.0
CUSP_HALF_MAX = 46.0

# How much shorter the total leader length has to get before a panel is allowed to
# change corners between clips. A near-tie must not be enough.
LAYOUT_MARGIN = 0.20


def find_cusps(g, cx, cy, r_sun):
    """The two ends of the LIT arc on the Sun's limb, read off the picture.

    The geometric version below is exact arithmetic on inexact inputs, and it is
    worst exactly where it is used. Two circles that nearly touch have an
    intersection that slides a long way along the limb for a small error in
    either centre or either radius, and both radii here are estimates while the
    Moon's centre comes from a fitted track carrying ~7 px of residual. Measured
    in the render, the box sat about 19 superpixel px off the visible horn.

    The Sun's limb, by contrast, is the best-measured thing in the frame - the
    fixed-radius fit is good to a few tenths of a pixel. So walk that circle,
    find where the brightness drops as the Moon's silhouette cuts across it, and
    take the two ends of the lit run. That uses no Moon centre and no lunar
    radius, and the answer is where the picture says the horn ends.
    """
    h, w = g.shape
    th = np.arange(CUSP_SAMPLES)*(2*math.pi/CUSP_SAMPLES)
    v = None
    band = []
    for rs in np.linspace(CUSP_R_LO, CUSP_R_HI, CUSP_R_STEPS)*r_sun:
        xs, ys = cx + rs*np.cos(th), cy + rs*np.sin(th)
        if xs.min() < 1 or ys.min() < 1 or xs.max() > w - 2 or ys.max() > h - 2:
            return None
        x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
        fx, fy = xs - x0, ys - y0
        s = (g[y0, x0]*(1 - fx)*(1 - fy) + g[y0, x0 + 1]*fx*(1 - fy)
             + g[y0 + 1, x0]*(1 - fx)*fy + g[y0 + 1, x0 + 1]*fx*fy)
        band.append(s)
        v = s if v is None else np.maximum(v, s)
    band = np.array(band)

    lit = v > CUSP_EDGE_FRAC*np.percentile(v, 95)
    if lit.all() or not lit.any():
        return None                  # no overlap at all, or no photosphere left

    # Longest run of lit samples, taken circularly.
    idx = np.nonzero(~lit)[0]
    gaps = np.diff(np.concatenate([idx, idx[:1] + CUSP_SAMPLES])) - 1
    k = int(np.argmax(gaps))
    if gaps[k] < CUSP_MIN_ARC:
        return None
    a = (idx[k] + 1) % CUSP_SAMPLES                    # first lit sample
    b = (idx[k] + gaps[k]) % CUSP_SAMPLES              # last lit sample

    def edge(i, step):
        """Sub-sample position of the lit/dark crossing just outside index i."""
        j = (i - step) % CUSP_SAMPLES
        v0, v1 = v[j], v[i]
        thr = CUSP_EDGE_FRAC*np.percentile(v, 95)
        f = 0.0 if v1 == v0 else (thr - v0)/(v1 - v0)
        t = th[j] + step*(2*math.pi/CUSP_SAMPLES)*min(max(f, 0.0), 1.0)
        return (cx + r_sun*math.cos(t), cy + r_sun*math.sin(t))

    # Middle of the lit run, which is the direction "into the crescent" from
    # either horn. The panels are pushed that way so the horn fills its box.
    mid = (a + (gaps[k] - 1)/2.0) % CUSP_SAMPLES

    """
    How BLUNT is the horn, in arc length.

    A fixed box suits a needle and not a wedge. Near second contact the two limbs
    cross almost tangentially and the horn is a hair; near first and last contact
    they cross at a wide angle and it is a blunt corner, where a box sized for a
    needle shows photosphere and two dark corners with no sense of convergence.

    Measured, not modelled, and free: the band scan above already samples several
    radii per angle, so counting how many of them clear the threshold gives the
    crescent's radial THICKNESS at that angle. Walking inward from a horn until
    that thickness reaches half the band gives the arc distance over which the
    horn opens up - small for a wedge, large for a needle - which is exactly the
    length the box needs to span. Thickness itself would saturate, since the band
    is only 0.08 R deep and a fat crescent fills it everywhere; the DISTANCE to
    reach a given thickness does not.
    """
    thick = (band > CUSP_EDGE_FRAC*np.percentile(v, 95)).sum(axis=0)
    want = CUSP_THICK_FRAC*CUSP_R_STEPS
    per = r_sun*(2*math.pi/CUSP_SAMPLES)        # arc length of one sample
    runlen = int(gaps[k])

    def opens_at(start, step):
        for j in range(runlen):
            if thick[(start + step*j) % CUSP_SAMPLES] >= want:
                return j*per
        return runlen*per

    return (edge(a, 1), edge(b, -1), mid*(2*math.pi/CUSP_SAMPLES),
            0.5*(opens_at(a, 1) + opens_at(b, -1)))


def fit_cusp_track(rows, deg):
    """Fit each cusp's angle across one capture and hand back the smooth version.

    Even at 0.2-1.9 px the per-frame answers still scatter, and magnified fivefold
    in the cusp panel 3 px is 30 px of visible wobble - the first cut of this had
    the tip leaving its own box. The motion underneath is not noisy: each cusp is
    driven by two circles sliding past each other at a constant rate, so its angle
    about the Sun's centre is a smooth, slowly curving function of time. Fitting it
    and regenerating is smooth by construction while keeping the accuracy that
    reading the picture bought.

    The two are tracked SEPARATELY by their stable upper/lower identity rather than
    as a bisector plus a half-separation. The pair passes through diametrically
    opposite when the centres are sqrt(rMoon^2 - rSun^2) = 86 px apart, which
    happens partway through this eclipse, and a bisector flips by half a turn
    exactly there. Per-cusp angles have no such singularity.

    A whole-eclipse geometric fit was tried and abandoned: solving for the Moon's
    radius and track against all 2328 measurements at once ought to beat this, but
    unbounded the radius ran to 279814 px (a clipped arccos is FLAT, so the solver
    loses its gradient) and even bounded the prediction started 123 px from
    measurements that agree with each other to 3 px. Do not retry it without first
    explaining that 123 px.
    """
    ts = [r[0] for r in rows]
    t0 = sum(ts)/len(ts)
    ts = [t - t0 for t in ts]              # centre, or the fit is ill-conditioned
    out = [None]*len(rows)
    dev = []
    for which in (1, 2):
        vs = [r[which] for r in rows]
        for i in range(1, len(vs)):        # unwrap
            while vs[i] - vs[i-1] > math.pi:
                vs[i] -= 2*math.pi
            while vs[i] - vs[i-1] < -math.pi:
                vs[i] += 2*math.pi
        keep = list(range(len(vs)))
        c = None
        for _ in range(4):
            if len(keep) < deg + 2:
                break
            c = np.polyfit([ts[i] for i in keep], [vs[i] for i in keep], deg)
            res = [abs(vs[i] - np.polyval(c, ts[i])) for i in range(len(vs))]
            s = sorted(res[i] for i in keep)
            cut = max(2.5*s[len(s)//2], 1e-4)
            nk = [i for i in keep if res[i] <= cut]
            if len(nk) < max(deg + 2, 0.4*len(keep)):
                break
            keep = nk
        if c is None:
            return None, None
        for i in range(len(rows)):
            th = float(np.polyval(c, ts[i]))
            dev.append(abs(math.atan2(math.sin(vs[i] - th), math.cos(vs[i] - th))))
            out[i] = (th,) if out[i] is None else (out[i][0], th)
    return out, dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="S:/solar-eclipse/out")
    ap.add_argument("--data", default="S:/solar-eclipse/Sun")
    ap.add_argument("--zoom", type=float, default=ZOOM)
    ap.add_argument("--cusp-zoom", type=float, default=CUSP_ZOOM)
    ap.add_argument("--panel", type=int, default=PANEL_PX)
    args = ap.parse_args()

    cfgpath = os.path.join(args.out, "configs", "timelapse.json")
    cfg = json.load(open(cfgpath))
    det = json.load(open(os.path.join(args.out, "diag", "centres.json")))
    man = json.load(open(os.path.join(args.out, "segments.json")))

    r_sun = det.get("rSun") or 279.0
    r_moon = det.get("rMoon") or 292.0
    fps_of = {f["name"]: f["fps"] for f in man["files"]}
    t0 = {f["name"]: datetime.datetime.fromisoformat(f["t0_utc"]) for f in man["files"]}
    epoch = min(t0.values())

    frames = cfg["frames"]
    for f in frames:
        f["utc"] = ((t0[f["file"]] - epoch).total_seconds()
                    + f["index"] / fps_of[f["file"]])

    tot = [f["utc"] for f in frames if f["state"] == "unfiltered"]
    t_mid = (min(tot) + max(tot)) / 2 if tot else 0.0

    """
    Moon position: MEASURED, not taken from drift.json.

    The first attempt used the differential rate corona-drift.js measured, applied
    as Moon = Sun + v*(t - t_mid). It put the Moon down-left of the Sun when the
    picture plainly shows it up-right - the sign convention of that measurement is
    the shift needed to ALIGN two frames, not the direction the Moon travelled.
    Inside smooth_track.py the same term is worth at most 9 px so the error was
    invisible there; here it is worth 240 px and threw every cusp into empty sky.

    Fitting the terminator instead removes the dependency altogether, and the
    residual below is a check that could not exist with a borrowed constant.
    """
    partials0 = [f for f in frames if f["state"] == "filtered"]
    mvx, mvy, nfit, resid = fit_moon_track(args.data, partials0, r_sun, r_moon,
                                           MOON_TRY_FRAMES)
    print("moon track %+0.4f, %+0.4f superpixel/s from %d frames, residual %.1f px"
          % (mvx[0], mvy[0], nfit, resid))

    def moon_of(f):
        return (f["cx"] + np.polyval(mvx, f["utc"]),
                f["cy"] + np.polyval(mvy, f["utc"]))

    # Sunspot, from the first filtered frame where the Moon covers least.
    partials = [f for f in frames if f["state"] == "filtered"]
    ox, oy, depth, seed = best_sunspot(args.data, partials, r_sun, SPOT_TRY_FRAMES)
    print("sunspot at Sun%+.1f%+.1f superpixel (r/R = %.2f), %.0f%% of the mean at "
          "that radius, from %s f%d"
          % (ox, oy, math.hypot(ox, oy)/r_sun, 100*depth, seed["file"], seed["index"]))

    """
    Prominences: detected once per EXPOSURE LEVEL, not once for all of totality.

    The first version measured them on the frame nearest mid-totality and held
    those offsets for every totality frame. That is wrong because totality was not
    shot at one exposure: the operator rode it from a short prominence exposure
    right after second contact out to a long one for the outer corona, and a
    prominence that stands clear at 1/500 s is buried in coronal glare at 1/15 s
    while a fainter one further out only appears in the long frames. Held offsets
    therefore pointed at whatever had been visible in a DIFFERENT exposure, which
    is how a panel came to sit on blank inner corona while an obvious prominence
    a quarter turn away had no box on it at all.

    A level is one constant-exposure run, keyed the same way the detector's
    level-bias correction keys it: file plus gain. Each is measured on its own
    middle frame, which costs one extra frame read per level.
    """
    unf = [f for f in frames if f["state"] == "unfiltered"]
    levels = {}
    for f in unf:
        levels.setdefault((f["file"], round(f["gain"], 4)), []).append(f)
    pang = {}
    for key in sorted(levels, key=lambda k: levels[k][0]["utc"]):
        fs = levels[key]
        mid = fs[len(fs)//2]
        mx, my = moon_of(mid)
        pang[key] = find_prominences(args.data, mid, mx, my, r_moon, 4)
        print("  level %-14s gain %-9s %3d frm -> %d prominence(s)  %s"
              % (key[0], key[1], len(fs), len(pang[key]),
                 ", ".join("%+.0f%+.0f r=%.2f snr=%.0f"
                           % (p[0], p[1], math.hypot(p[0], p[1])/r_moon, p[2])
                           for p in pang[key])))

    """
    Send each feature to its nearest corner, not to a fixed one.

    With a fixed mapping the leader lines cross the frame diagonally - the cusps
    sweep right around the limb over 46 minutes, so any corner assigned to one is
    wrong for half the video. Choosing the assignment that minimises total line
    length is a small problem, solved by trying every way of putting k features in
    4 corners.

    Changing it every frame would make the panels flicker between subjects, so a
    new assignment has to win for HOLD_FRAMES consecutive frames before it is
    adopted. Features move slowly and smoothly, so in practice this switches a
    handful of times, at the moments where a different corner genuinely is nearer.

    The hold is abandoned outright when the feature LIST changes rather than
    moves. It smooths a drifting optimum; it must not carry an assignment across a
    discontinuity, and there are two: the number of features changing, and a new
    totality exposure level, whose prominences are detected separately and ranked
    by their own strengths. A held permutation maps slot number to corner, so
    after a re-ranking it sends slot 1 wherever slot 1 used to go and the panels
    end up pointing across each other. Measured on 14_14_36 f340 the held layout
    cost 2377 px of leader line against 1875 for this frame's own optimum, and the
    two extra hundreds were two lines crossing over the middle of the corona.

    Minimising total length is also what keeps the leaders from crossing at all: a
    minimum-length matching cannot contain a crossing pair, since swapping any two
    crossing assignments is strictly shorter by the triangle inequality.
    """
    import itertools
    HOLD_FRAMES = 9
    perms = {k: list(itertools.permutations(range(4), k)) for k in range(1, 5)}
    ow, oh = cfg["outW"], cfg["outH"]
    corner_xy = [(0.0, 0.0), (0.0, oh), (ow, 0.0), (ow, oh)]
    cur_perm = None
    pend, pend_n = None, 0
    prev_ident = None

    """
    The cusps are named by where they are in the picture, not by which leads.

    Leading and trailing would be the better names if they could be measured, and
    they cannot be. The cusps are the two intersections of the limbs, so they sit
    symmetrically either side of the Sun-Moon line - and the Moon travels very
    nearly along that line, which makes their projections onto its direction equal
    to first order. Measured over the 1165 frames that have both: 12.7 px of
    separation along the motion, 3.4 px at worst, against 7.1 px of residual in
    the Moon track itself. The choice was noise, and it did quietly swap the two
    horns once during the video.

    In image y they are 466 px apart at the median and never closer than 247, and
    the sense never changes: the Moon crosses nearly horizontally, so the horns
    stay one above the other from first contact to last. Upper and lower are also
    what a viewer can check for themselves, which leading and trailing are not
    without knowing the sky orientation - and nothing here plate-solves, so that
    orientation is not known.
    """
    """
    Measure the cusps on every filtered frame, then fit each capture.

    A pre-pass, because the fit needs the whole capture before any frame can be
    placed. Cost is one frame read per filtered frame.
    """
    ser_cache = [None, None]

    def plane_of(f):
        if ser_cache[0] != f["file"]:
            if ser_cache[1] is not None:
                ser_cache[1].close()
            ser_cache[1] = SerFile(os.path.join(args.data, f["file"]))
            ser_cache[0] = f["file"]
        return green_plane(ser_cache[1], f["index"])

    raw, mid_of, open_of = {}, {}, {}
    for f in frames:
        if f["state"] != "filtered":
            continue
        cu = find_cusps(plane_of(f), f["cx"], f["cy"], r_sun)
        if cu is None:
            continue
        a, b, thmid, opens = cu
        if a[1] > b[1]:                     # upper first, by image position
            a, b = b, a
        raw.setdefault(f["file"], []).append(
            (f["index"],
             math.atan2(a[1] - f["cy"], a[0] - f["cx"]),
             math.atan2(b[1] - f["cy"], b[0] - f["cx"])))
        mid_of[(f["file"], f["index"])] = thmid
        open_of.setdefault(f["file"], []).append(opens)
    if ser_cache[1] is not None:
        ser_cache[1].close()

    cusp_of, cusp_box = {}, {}
    for fn in sorted(raw):
        rows = sorted(raw[fn])
        if len(rows) < CUSP_MIN_FRAMES:
            print("  %-14s only %d measurements - no cusp panels"
                  % (fn, len(rows)))
            continue
        # Quadratic where there is enough data. The cusps sweep faster as totality
        # nears, so a straight line describes worst exactly the captures whose own
        # measurements are weakest.
        deg = 2 if len(rows) >= CUSP_QUAD_MIN else 1
        fitted, dev = fit_cusp_track(rows, deg)
        if fitted is None:
            continue
        for (i, _1, _2), ths in zip(rows, fitted):
            cusp_of[(fn, i)] = ths
        # One box size per CAPTURE, from the median opening length. Both horns of
        # a crescent are the same shape by symmetry, and the wedge angle changes
        # far too slowly to be worth re-deciding per frame.
        op = sorted(open_of[fn])
        half = min(max(CUSP_BOX_K*op[len(op)//2], CUSP_HALF_MIN), CUSP_HALF_MAX)
        cusp_box[fn] = half
        dev = sorted(d*r_sun for d in dev)
        print("  %-14s %3d frames, deg %d, scatter %.2f px median; horn opens over"
              " %.0f px -> half-box %.0f px (zoom %.1fx)"
              % (fn, len(rows), deg, dev[len(dev)//2], op[len(op)//2], half,
                 args.panel/(4.0*half)))

    box = args.panel / args.zoom / 2.0   # half-width of the source box, FINE px
    halfsp = box / 2.0                   # ... in superpixel units
    n_cusp = 0
    counts = {}
    per_frame = []
    for f in frames:
        mx, my = moon_of(f)

        """
        Emit only the features that exist in THIS frame.

        The old version always produced four, padding with duplicates whenever the
        cusps did not exist, so two panels magnified the same thing and the labels
        would have lied about it. Every feature here is now gated on its own
        subject actually being there to look at:

          sunspot      until the Moon covers it, and again after it clears
          lunar limb   only while the Moon overlaps the disc
          cusps        only while the limbs genuinely intersect
          prominence   only where the level's own search found one (see above)
        """
        feats = []
        if f["state"] == "unfiltered":
            for dxp, dyp, _snr in pang.get((f["file"], round(f["gain"], 4)), []):
                feats.append((mx + dxp, my + dyp, "prominence", args.zoom))
        else:
            spot = (f["cx"] + ox, f["cy"] + oy)
            if math.hypot(spot[0] - mx, spot[1] - my) > r_moon:
                feats.append((spot[0], spot[1], "sunspot", args.zoom))
            dx, dy = f["cx"] - mx, f["cy"] - my
            dd = math.hypot(dx, dy) or 1.0
            if abs(r_moon - r_sun) < dd < r_sun + r_moon:
                # Deepest incursion: the Moon's limb point nearest the Sun's
                # centre. Only meaningful while that point is on the photosphere.
                feats.append((mx + r_moon*dx/dd, my + r_moon*dy/dd,
                              "lunar limb", args.zoom))
            ths = cusp_of.get((f["file"], f["index"]))
            if ths:
                n_cusp += 1
                tm = mid_of.get((f["file"], f["index"]))
                half = cusp_box[f["file"]]
                czoom = args.panel/(4.0*half)
                shift = CUSP_INSET_FRAC*half/r_sun
                pts = []
                for t in ths:
                    # Slide the box back along the limb, into the crescent, so the
                    # horn fills it instead of ending in the middle of empty sky.
                    if tm is not None:
                        d = math.atan2(math.sin(tm - t), math.cos(tm - t))
                        t += math.copysign(shift, d)
                    pts.append((f["cx"] + r_sun*math.cos(t),
                                f["cy"] + r_sun*math.sin(t)))
                pts.sort(key=lambda p: p[1])
                feats.append((pts[0][0], pts[0][1], "upper cusp", czoom))
                feats.append((pts[1][0], pts[1][1], "lower cusp", czoom))

        feats = feats[:4]
        counts[len(feats)] = counts.get(len(feats), 0) + 1
        per_frame.append((f, feats))
        del f["utc"]

    """
    Decide the layout ONCE PER CLIP, not per frame.

    A per-frame optimum with a hold was switching panels between corners in the
    middle of a capture, and worst at the very start, where the first two runs are
    only 28 and 34 frames long and are separated by eight and a half minutes - so
    the geometry jumps between them and the best assignment jumps with it. A layout
    that changes while a clip is playing reads as the boxes swapping targets.

    The layout is a property of the clip, so it is chosen from the clip's MEDIAN
    feature positions and held for every frame of it. It can then only change where
    a change is already expected and hidden: at a capture boundary, under the
    dissolve. Keyed on the label list as well as the file, so a capture that loses
    a feature partway - the sunspot going behind the Moon inside 14_01_54 - gets a
    fresh layout for the frames after it rather than an assignment built for a
    feature that is no longer there.
    """
    groups, order = {}, []
    for j, (f, feats) in enumerate(per_frame):
        if not feats:
            continue
        key = (f["file"], tuple(t[2] for t in feats))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((f, feats))

    """
    A panel NEVER changes corner unless its own subject is new.

    Optimising each clip on its own kept trading panels around, and weighting
    continuity was not enough either, because the trades were not caused by
    near-ties in the first place. What actually moved them was a feature LEAVING:
    when the sunspot goes behind the Moon partway through 14_01_54 it frees the
    upper-left corner, the unconstrained optimum then pulls everyone else into it,
    and two panels jump mid-clip. The same happens in reverse when a feature comes
    back. Nothing about the remaining features changed - only the vacancy.

    So corners are INHERITED. Each feature takes the corner of the nearest feature
    of the same name in the previous clip; only features with no predecessor get
    placed, into whatever corners are still free, by shortest leader. Inheriting by
    nearest position rather than by rank matters for the prominences, which are
    re-detected and re-ranked at every exposure level - by rank, slot 2 of one
    level is a different prominence from slot 2 of the next and the panels would
    shuffle even though the picture had barely changed.

    The memory is cleared when the STATE changes, because totality has an entirely
    different cast (four prominences against a sunspot, a limb and two cusps) and
    the partial phases either side of it are separated by the largest gap in the
    video. Re-deciding there is expected, and hidden by it.
    """
    layout = {}
    prev = []                 # (x, y, label, corner) from the previous clip
    prev_state = None
    n_switch = 0
    for key in order:
        rows = groups[key]
        n = len(rows[0][1])
        labels = list(key[1])
        state = rows[0][0]["state"]
        if state != prev_state:
            prev = []
        prev_state = state

        med = []
        for i in range(n):
            xs = sorted(t[1][i][0] - (t[0]["cx"] - ow/2.0) for t in rows)
            ys = sorted(t[1][i][1] - (t[0]["cy"] - oh/2.0) for t in rows)
            med.append((xs[len(xs)//2], ys[len(ys)//2]))

        assigned = [None]*n
        used, taken = set(), set()
        # Nearest-predecessor inheritance, closest pairs first so the obvious
        # matches are made before the ambiguous ones.
        pairs = sorted(
            ((math.hypot(med[i][0] - p[0], med[i][1] - p[1]), i, j)
             for i in range(n) for j, p in enumerate(prev) if p[2] == labels[i]))
        for _d, i, j in pairs:
            if assigned[i] is not None or j in taken or prev[j][3] in used:
                continue
            assigned[i] = prev[j][3]
            used.add(prev[j][3])
            taken.add(j)

        rest = [i for i in range(n) if assigned[i] is None]
        if rest:
            free = [c for c in range(4) if c not in used]
            pick = min(itertools.permutations(free, len(rest)),
                       key=lambda pm: sum(
                           math.hypot(med[rest[j]][0] - corner_xy[pm[j]][0],
                                      med[rest[j]][1] - corner_xy[pm[j]][1])
                           for j in range(len(rest))))
            for j, i in enumerate(rest):
                assigned[i] = pick[j]
            n_switch += 1
            print("  %-14s placed %s"
                  % (key[0], ", ".join(labels[i] for i in rest)))

        layout[key] = tuple(assigned)
        prev = [(med[i][0], med[i][1], labels[i], assigned[i]) for i in range(n)]

    for f, feats in per_frame:
        if not feats:
            f["insets"] = []
            continue
        pm = layout[(f["file"], tuple(t[2] for t in feats))]
        # Each inset names its own corner (0 upper-left, 1 lower-left, 2 upper-
        # right, 3 lower-right) because there is no longer one per corner. Centres
        # are in SUPERPIXEL coordinates, the same units as the disc track, so the
        # renderer scales them by the drizzle factor exactly as it does the crop.
        f["insets"] = [{"cx": round(feats[i][0], 2), "cy": round(feats[i][1], 2),
                        "corner": pm[i], "label": feats[i][2],
                        "zoom": round(feats[i][3], 3)}
                       for i in range(len(feats))]

    cfg["insetPanel"] = args.panel
    cfg["insetZoom"] = args.zoom
    cfg["insetBox"] = round(2 * halfsp, 2)
    json.dump(cfg, open(cfgpath, "w"))
    print("panel %d px at %.1fx (source box %.0f superpixel); cusps on %d of %d "
          "frames; %d of %d clips move a panel"
          % (args.panel, args.zoom, 2 * halfsp, n_cusp, len(frames), n_switch,
             len(layout)))
    print("panels per frame: %s"
          % ", ".join("%d on %d frames" % (k, counts[k]) for k in sorted(counts)))
    print("updated %s" % cfgpath)


if __name__ == "__main__":
    main()
