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

# Video frames a totality exposure level needs before prominences are detected on
# it rather than borrowed. Below this it is a point on the operator's ramp.
PROM_LEVEL_MIN = 8

# Terminator fits worse than this are not the Moon's limb and are discarded.
MOON_FIT_MAX_RMS = 12.0
MOON_TRY_FRAMES = 14


def tune(out_dir, log=None):
    """Resolve the panel geometry from the config against the surveyed radius.

    These are the constants that fail hardest on another camera: every one of
    them is a box size or a separation measured in pixels on a sensor where the
    disc happened to be 279 px across.
    """
    global PROM_MIN_SEP_PX, PROM_R_INNER, PROM_R_OUTER, PROM_MIN_SNR
    global PROM_LEVEL_MIN, SPOT_MAX_R, MIN_RING_PX, MOON_FIT_MAX_RMS
    global CUSP_HALF_MIN, CUSP_HALF_MAX, BEAD_HALF_MIN, BEAD_HALF_MAX
    global BEAD_MIN_AREA, BEAD_MAX_THICK, BEAD_ARC_MAX, BEAD_ARC_MIN
    global BEAD_R_INNER, BEAD_R_OUTER, BEAD_SAT, BEAD_NEAR_FRAMES, BEAD_RUN_GAP
    global ZOOM, PANEL_PX, USE_TOP_BOTTOM
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ecl.params import load

    P = load(out_dir, create=False)
    PROM_MIN_SEP_PX = P.px("panels.prom_sep_r")
    PROM_R_INNER = P.get("panels.prom_r_inner", PROM_R_INNER)
    PROM_R_OUTER = P.get("panels.prom_r_outer", PROM_R_OUTER)
    PROM_MIN_SNR = P.get("panels.prom_min_snr", PROM_MIN_SNR)
    PROM_LEVEL_MIN = P.get("panels.prom_level_min", PROM_LEVEL_MIN)
    SPOT_MAX_R = P.get("panels.spot_max_r", SPOT_MAX_R)
    MIN_RING_PX = max(4, int(P.px("panels.spot_ring_r")))
    MOON_FIT_MAX_RMS = P.px("panels.moon_fit_max_rms_r")
    CUSP_HALF_MIN, CUSP_HALF_MAX = P.px("panels.cusp_half_r")
    BEAD_HALF_MIN, BEAD_HALF_MAX = P.px("panels.bead_half_r")
    BEAD_MIN_AREA = max(1.0, P.area("beads.min_area_r2"))
    BEAD_MAX_THICK = P.px("beads.max_thick_r")
    BEAD_ARC_MAX = P.get("beads.arc_max_deg", BEAD_ARC_MAX)
    BEAD_ARC_MIN = P.get("beads.arc_min_deg", BEAD_ARC_MIN)
    BEAD_R_INNER, BEAD_R_OUTER = P.get("beads.annulus", [BEAD_R_INNER, BEAD_R_OUTER])
    BEAD_SAT = P.get("beads.sat", BEAD_SAT)
    BEAD_NEAR_FRAMES = P.get("beads.near_frames", BEAD_NEAR_FRAMES)
    BEAD_RUN_GAP = P.get("beads.run_gap", BEAD_RUN_GAP)
    ZOOM = P.get("panels.zoom", ZOOM)
    USE_TOP_BOTTOM = P.get("panels.use_top_bottom", USE_TOP_BOTTOM)
    if log:
        log(f"  tuned to r={P.radius_px:.0f}px: cusp box "
            f"{CUSP_HALF_MIN:.0f}-{CUSP_HALF_MAX:.0f}, bead box "
            f"{BEAD_HALF_MIN:.0f}-{BEAD_HALF_MAX:.0f}, prom separation "
            f"{PROM_MIN_SEP_PX:.0f} px")
    return P


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


"""
Baily's beads: photosphere through lunar valleys, found by where the sensor
clipped rather than by how bright something is.

Every other feature here is found by standing over its background. A bead cannot
be, because it is not merely brighter than the corona - it is 20 to 500 times
brighter, and the sensor clips solid across it. What the frame carries is not a
peak but a PLATEAU, and a plateau has no meaningful argmax: the brightest pixel
of a saturated region is wherever noise happened to land, which is why sampling
the maximum walked around the arc from frame to frame.

The shape of the clipped region is the measurement. Projected onto azimuth around
the lunar limb it is one closed 360 degree ring when the filter first comes off,
and it opens and shrinks monotonically to a 34 degree arc by the time the corona
exposure starts. That single number - the widest clipped arc - answers both
questions the panel needs: whether there is a bead, and whether it has shrunk
enough to be worth magnifying.

The second half of that matters more than it sounds. Pulling a panel's exposure
down recovers structure only where the sensor did NOT clip; over a plateau it
turns a white disc into a grey one and shows nothing. So while the arc is wide
there is genuinely nothing to look at, no matter how it is exposed, and the panel
is not emitted at all.
"""
BEAD_NAZ = 720                  # azimuth bins around the limb, 0.5 deg each
BEAD_R_INNER = 0.93
BEAD_R_OUTER = 1.05
BEAD_SAT = 0.90                 # fraction of full scale counted as clipped

"""
The panel is gated on the clipped region's THICKNESS, not on its arc.

Arc width was the obvious measure and it is the wrong one. Across the frames
where the picture changes completely - a solid white plateau resolving into a
chain of separate beads - the arc only goes from 49 degrees to 30, because the
beads that survive are spread along the same stretch of limb the plateau covered.
Gating on it admitted plateaus at any threshold that also admitted beads.

Radial thickness, the clipped area divided by the arc length it spans, separates
them completely over the same frames:

    f1124   arc 49.0   thick 24.00     flat white square
    f1142   arc 39.5   thick 13.56
    f1154   arc 36.5   thick  4.82     chain, knots visible
    f1192   arc 30.5   thick  1.22

which is the physical difference: a plateau is photosphere filling the whole
annulus, a bead chain is a thin line of it left between lunar peaks. Threshold
set from rendered panels - 24 is a white square, 5 is the picture.
"""
BEAD_MAX_THICK = 6.0

# Smallest clipped area, superpixel px, still worth calling a bead.
#
# Shared with `ecl.beadwindow`, which uses the same floor to decide the window
# the video dwells on. They must agree: with this at 150 and that at 40 the panel
# ran f1151-1202 while the dwell held f1189-1255, so the label came off partway
# through the very sequence it had been slowed down for.
#
# The run does not end when the beads do - the clipped region keeps shrinking
# smoothly, 1132 px at f1151 down to 9 px at f1296, long after the last bead has
# gone and only the chromosphere is still clipping in the long exposure. Without
# a floor the panel stayed up to f1300, magnifying a plain arc and calling it
# Baily's beads.
from ecl.beadwindow import MIN_AREA as BEAD_MIN_AREA  # noqa: E402
BEAD_ARC_MAX = 90.0             # loose: rejects the closed ring, nothing else
BEAD_ARC_MIN = 1.5              # under this it is a hot pixel, not a bead

# Panel box, superpixel px: one size for the whole run, from the median extent of
# the clipped chain, then clamped. The lower bound keeps a nearly-gone bead from
# being magnified past its own noise, the upper keeps the box from growing until
# it is no longer an inset.
BEAD_BOX_K = 1.35
BEAD_HALF_MIN = 45.0
BEAD_HALF_MAX = 80.0

# Measurements a bead run needs before its track is worth a quadratic.
BEAD_QUAD_MIN = 25

# Frames either side of a state change that count as "at a contact", and the
# largest hole a bead run may bridge. The gap allows for the odd frame where the
# arc briefly falls outside the width bounds without the run being two events.
BEAD_NEAR_FRAMES = 10
BEAD_RUN_GAP = 4


def find_bead_arc(g, mx, my, r_moon, max_value=65535.0):
    """Clipped limb region: (cx, cy, extent, arc deg, thickness, area).

    The CENTROID is what the panel is centred on, not the midpoint of the widest
    arc. Once the chain breaks into several beads, which one is widest changes
    from frame to frame, and a box centred on it teleports between them - measured
    at 40 to 45 px, alternating every frame or two. The centroid of the whole
    clipped set does not care how the set is partitioned, so it moves as smoothly
    as the beads themselves do. The widest arc is still returned, because that is
    what the "is it small enough to magnify" gate is written against.
    """
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - mx, yy - my)
    ann = ((rr > BEAD_R_INNER*r_moon) & (rr < BEAD_R_OUTER*r_moon)
           & (g >= BEAD_SAT*max_value))
    if not ann.any():
        return None
    ys, xs = yy[ann].astype(np.float64), xx[ann].astype(np.float64)
    area = float(xs.size)
    bx, by = float(xs.mean()), float(ys.mean())
    # p90 rather than max: a single hot pixel off the end of the chain should not
    # decide the box size for the whole run.
    ext = float(np.percentile(np.hypot(xs - bx, ys - by), 90))
    th = np.degrees(np.arctan2(yy[ann] - my, xx[ann] - mx)) % 360.0
    occ = np.zeros(BEAD_NAZ, bool)
    occ[np.minimum((th/(360.0/BEAD_NAZ)).astype(int), BEAD_NAZ - 1)] = True
    if occ.all():
        return (bx, by, ext, 360.0, area/(2*math.pi*r_moon), area)

    # Contiguous runs, on a circle: the wrap is a real case here, since the arc
    # sits wherever the last of the photosphere happens to be.
    d = np.diff(np.r_[occ[-1], occ].astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    if len(ends) and (not len(starts) or ends[0] < starts[0]):
        ends = np.r_[ends[1:], ends[0]]
    best = 0
    for s, e in zip(starts, ends):
        n = int((e - s) % BEAD_NAZ) or BEAD_NAZ
        best = max(best, n)
    if not best:
        return None
    deg = best*(360.0/BEAD_NAZ)
    # Thickness: clipped area over the arc length it covers, in superpixel px.
    return (bx, by, ext, deg, area/max(r_moon*math.radians(deg), 1.0), area)


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
# The floor was 14.0 and it was too low, measured 2026-08-16. At 15 superpixel px
# of arc from a near-tangential limb crossing the crescent is a sliver well under
# a pixel thick, so the panel faithfully showed a horn with no visible width:
# frame 0's upper-cusp panel was 1.2% lit against 100% for the sunspot and 72%
# for the lunar limb, and the same emptiness is in the PJSR render, so it is the
# sizing and not the port or the position. The horn is correctly located — the
# measured horn sits at 259.0 deg, the inset at 257.3, and the crescent's middle
# at 148, so the offset is the intended CUSP_INSET_FRAC push in the right
# direction. Widening the box is what makes it legible: 1.2% lit at the old
# floor, 11.0% at a half-box of 42, where the horn reads as a clean taper.
CUSP_HALF_MIN = 40.0
CUSP_HALF_MAX = 46.0

# How much shorter the total leader length has to get before a panel is allowed to
# change corners between clips. A near-tie must not be enough.
LAYOUT_MARGIN = 0.20

# Panel inset from the frame edge, FINE px - must match draw_insets.
PANEL_MARGIN = 24
# Below this a panel is too small to read; six slots are used instead.
PANEL_MIN_PX = 200
USE_TOP_BOTTOM = True

# Pixels of extra leader length worth paying per radian of extra spacing between
# panels. Length alone always bunches: the shortest leaders are the ones that
# never leave the crowded side of the frame, so with four prominences on the left
# of the disc all four panels stack down the left edge.
#
# Swept over the whole video, counting frames where three or more panels end up
# on one half of the frame:
#
#   weight      0    400   1000   2000   4000   8000
#   packed   1157   1086    660    342    342    342
#   min gap    37     37     74     74     74     74   degrees
#   leader     560    570    715    728    728    728   px at p90
#
# It saturates at 2000 - past that the spread term dominates every comparison it
# can win and nothing further changes. The residue of 342 is not a tuning
# failure: a three-panel clip has no balanced arrangement across six slots, and
# some of those genuinely belong on one side.
SPREAD_WEIGHT = 2000.0


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
    tune(args.out, log=print)

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

    A level has to be long enough to BE a level, though.

    Keying on gain splits a run wherever the exposure changed, which is right when
    the operator settled somewhere and wrong while they were still moving. The
    start of 14_14_36 is five "levels" of 4, 2, 1, 1 and 1 frames inside ten video
    frames - a ramp, not five exposures - and each one got its own detection, its
    own four prominences and its own ranking. The panels moved 380 to 560 px
    between consecutive frames there. A level under PROM_LEVEL_MIN borrows the
    detection from the nearest level in the same capture that clears the bar, so a
    ramp shows the prominences of the exposure it is heading for.

    Resolve frames are left out entirely: they show no prominence panels, so
    detecting on them is 82 frame reads for results nothing consumes.
    """
    unf = [f for f in frames
           if f["state"] == "unfiltered" and not f.get("resolve")]
    levels = {}
    for f in unf:
        levels.setdefault((f["file"], round(f["gain"], 4)), []).append(f)

    order_l = sorted(levels, key=lambda k: levels[k][0]["utc"])
    solid = [k for k in order_l if len(levels[k]) >= PROM_LEVEL_MIN] or order_l
    pang, borrow = {}, {}
    for key in order_l:
        if key in solid:
            continue
        # Nearest solid level in time, preferring the same capture.
        t = levels[key][0]["utc"]
        same = [k for k in solid if k[0] == key[0]] or solid
        borrow[key] = min(same, key=lambda k: abs(levels[k][0]["utc"] - t))

    for key in solid:
        fs = levels[key]
        mid = fs[len(fs)//2]
        mx, my = moon_of(mid)
        pang[key] = find_prominences(args.data, mid, mx, my, r_moon, 4)
        print("  level %-14s gain %-9s %3d frm -> %d prominence(s)  %s"
              % (key[0], key[1], len(fs), len(pang[key]),
                 ", ".join("%+.0f%+.0f r=%.2f snr=%.0f"
                           % (p[0], p[1], math.hypot(p[0], p[1])/r_moon, p[2])
                           for p in pang[key])))
    for key, src in sorted(borrow.items(), key=lambda kv: levels[kv[0]][0]["utc"]):
        pang[key] = pang[src]
        print("  level %-14s gain %-9s %3d frm -> borrows gain %s (too short to "
              "detect on)" % (key[0], key[1], len(levels[key]), src[1]))

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
    """
    EIGHT slots, not four: the corners plus a midpoint on each edge.

    With only corners, a feature near the middle of an edge has no slot anywhere
    near it and has to reach to a corner, which drags its leader diagonally across
    the frame at a steep angle - the panel points down the side of the picture
    while its subject sits beside it. Four features and four corners also leaves no
    slack at all: whatever the geometry, every corner is spoken for.

    The extra four are not extra panels. Only the features that exist get drawn, so
    at four features and eight slots half the slots stay empty and each feature
    takes whichever is nearest - which is the whole point. It also gives the
    crossing penalty room to work, since there is now usually a slot on the same
    side as the subject.

    Eight slots where the frame has room, six where it does not.

    A panel may not sit on the disc - it would hide the very thing it points at,
    and a bottom-centre panel once covered the bead chain it was magnifying. The
    disc is round and the frame is not, so there is far more clear space beside
    it than above it: 616 px against 316 in this window. A full-size panel fits
    the sides and not the top.

    So the panel is SHRUNK to whatever clears the disc when the top and bottom
    are wanted - 292 px instead of 420 here, 70% of the edge - and that buys the
    whole perimeter to spread panels around instead of three slots a side. Set
    panels.use_top_bottom = false to keep the bigger panels instead.

    Order matters and is shared with `tl_render.draw_insets`: 0-3 the corners
    (upper-left, lower-left, upper-right, lower-right), 4 left, 5 right, then
    6 top and 7 bottom.
    """
    ow, oh = cfg["outW"], cfg["outH"]
    corner_xy = [(0.0, 0.0), (0.0, oh), (ow, 0.0), (ow, oh),
                 (0.0, oh / 2), (ow, oh / 2)]
    if USE_TOP_BOTTOM:
        # Largest panel whose top/bottom placement still clears the disc, in
        # FINE px; the slot list only grows if one actually fits.
        fit = int((2 * oh - 2 * 2 * r_moon) / 2) - PANEL_MARGIN
        if fit >= PANEL_MIN_PX:
            if fit < args.panel:
                print("  panel %d -> %d px so the top and bottom slots clear "
                      "the disc" % (args.panel, fit))
                args.panel = fit
            corner_xy += [(ow / 2, 0.0), (ow / 2, oh)]
        else:
            print("  no room for top/bottom panels (%d px clear); "
                  "using six slots" % max(fit, 0))
    perms = {k: list(itertools.permutations(range(len(corner_xy)), k))
             for k in range(1, 5)}

    """
    A leader must not be routed across the Moon.

    Corners were chosen on leader LENGTH alone, and length does not know what it
    is drawing over. Measured on the last cut, a leader passed inside the lunar
    disc on 1891 prominence frames, 1169 lunar-limb frames and most of the cusp
    frames - one of them within 3 px of the disc centre, so the line ran the full
    diameter of the subject.

    The disc is the darkest thing in the frame and a hairline over it is far more
    visible than the same line over corona, which is why this reads worse than the
    length numbers suggest. Crossing is priced rather than forbidden: with four
    features spread around a limb and four fixed corners there is not always a
    clean assignment, and a crossing leader still beats no leader. The penalty is
    simply larger than any leader can be, so length only breaks ties among
    assignments that cross equally often.
    """
    disc = (ow / 2.0, oh / 2.0)

    def over_disc(fx, fy, cx, cy, r):
        """Does the segment (fx,fy)-(cx,cy) pass within r of the disc centre?"""
        dx, dy = cx - fx, cy - fy
        L = dx*dx + dy*dy
        t = 0.0 if L == 0 else max(0.0, min(1.0, ((disc[0] - fx)*dx
                                                  + (disc[1] - fy)*dy) / L))
        return math.hypot(fx + t*dx - disc[0], fy + t*dy - disc[1]) < r

    CROSS_PENALTY = 10_000.0

    def _side(a, b, c):
        return (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])

    def leaders_cross(p1, p2, q1, q2):
        """Do segments p1-p2 and q1-q2 properly cross?"""
        d1, d2 = _side(q1, q2, p1), _side(q1, q2, p2)
        d3, d4 = _side(p1, p2, q1), _side(p1, p2, q2)
        return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))

    def leader_cost(fx, fy, corner, r):
        d = math.hypot(fx - corner[0], fy - corner[1])
        return d + (CROSS_PENALTY if over_disc(fx, fy, corner[0], corner[1], r)
                    else 0.0)
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

    """
    Beads, on every totality frame - then kept only where they connect to a
    contact.

    Measuring the clipped arc finds real beads and also finds something else: at
    the long exposures mid-totality the inner corona and chromosphere clip too,
    over an arc of 21 to 30 degrees, which passes a width test cleanly. Nine
    frames of 14_14_36 came back labelled as beads that way, three minutes from
    any photosphere.

    Colour does not separate them - both are near-white where they clip, R/G 0.94
    against 0.98 - and a contact-time window is both an ephemeris dependency and
    too tight, since the beads at second contact run on for fifty frames past the
    boundary while the corona exposure is coming up.

    What does separate them is CONTINUITY. Beads are the end of a continuous
    process that starts at the filter change, so the frames carrying them form one
    unbroken run reaching back to it. A clipped corona mid-totality forms its own
    run touching nothing. So candidates are grouped into runs and a run is kept
    only if it contains a frame that is either part of the second-contact resolve
    or sits within BEAD_NEAR_FRAMES of a state change - the same structural marker
    the stacking rule uses, and no clock.
    """
    cand, order_i, bead_seen, bead_wide = {}, [], 0, 0
    for j, f in enumerate(frames):
        if f["state"] != "unfiltered":
            continue
        # The annulus goes on the config's OWN disc centre, not on moon_of.
        #
        # Through totality the detector measures the Moon directly - ring search
        # on the corona, then per-frame corona correlation - so cx/cy IS the Moon
        # here. moon_of adds the terminator track, which is fitted on the partial
        # phases from ten frames at 7.2 px residual and then extrapolated, and it
        # sits a steady 12 px away.
        #
        # Twelve pixels is fatal for this measurement specifically. The annulus is
        # only 0.12 r_moon thick, so an offset that size puts it inside the limb on
        # one side and outside on the other, and as the chain shortens and slides
        # the fraction of it caught by the misplaced ring keeps changing. That drags
        # the centroid: measured on moon_of it walks 640 -> 607 over the run, while
        # on the disc centre it holds at 643 -> 649. The box looked on target and
        # then left, which is exactly what the drift produces. The prominences keep
        # moon_of - their annulus is 0.35 r_moon thick and shrugs the offset off.
        arc = find_bead_arc(plane_of(f), f["cx"], f["cy"], r_moon)
        if arc is None:
            continue
        bead_seen += 1
        bx, by, ext, deg, thick, area = arc
        if (deg > BEAD_ARC_MAX or deg < BEAD_ARC_MIN or thick > BEAD_MAX_THICK
                or area < BEAD_MIN_AREA):
            bead_wide += 1
            continue
        cand[j] = (bx, by, ext)
        order_i.append(j)

    near = set()
    for j, f in enumerate(frames):
        if f.get("resolve"):
            near.add(j)
            continue
        lo, hi = max(0, j - BEAD_NEAR_FRAMES), min(len(frames), j + BEAD_NEAR_FRAMES + 1)
        if any(frames[k]["state"] != f["state"] for k in range(lo, hi)):
            near.add(j)

    bead_of, n_runs, n_drop = {}, 0, 0
    run = []
    for j in order_i + [None]:
        if run and (j is None or j - run[-1] > BEAD_RUN_GAP):
            if any(k in near for k in run):
                n_runs += 1
                """
                One box size for the whole run, and a MODELLED centre - the same
                two rules the cusps are placed by, and for the same reasons.

                Sizing the box per frame from that frame's own extent re-decides
                the magnification 87 times: measured, the zoom walked 1.167 to
                2.100 in per-frame steps with a 0.755 jump in the middle of it,
                which is the panel breathing. The beads shrink far too slowly for
                a per-frame size to be telling the truth about anything.

                The centre is fitted against the RAW frame index rather than
                smoothed over the frame list, because the prominence level repeats
                each raw frame twice to hold it on screen - a moving average over
                the list would weight those doubled frames twice and pull the fit
                toward them. Degree 2: the run is a few seconds long and the beads
                slide along the limb at a rate that is visibly not constant.
                """
                rows = sorted((frames[k]["index"], cand[k][0], cand[k][1]) for k in run)
                ext = sorted(cand[k][2] for k in run)
                half = min(max(BEAD_BOX_K*ext[len(ext)//2], BEAD_HALF_MIN),
                           BEAD_HALF_MAX)
                deg_fit = 2 if len(rows) >= BEAD_QUAD_MIN else 1
                ts = [r[0] for r in rows]
                px = np.polyfit(ts, [r[1] for r in rows], deg_fit)
                py = np.polyfit(ts, [r[2] for r in rows], deg_fit)
                dev = sorted(math.hypot(np.polyval(px, r[0]) - r[1],
                                        np.polyval(py, r[0]) - r[2]) for r in rows)
                print("  beads: %s f%d-%d, %d frames, deg %d, box half %.0f px "
                      "(zoom %.2fx), track scatter %.1f px median"
                      % (frames[run[0]]["file"], rows[0][0], rows[-1][0],
                         len(run), deg_fit, half, args.panel/(4.0*half),
                         dev[len(dev)//2]))
                for k in run:
                    t = frames[k]["index"]
                    bead_of[(frames[k]["file"], t)] = (
                        float(np.polyval(px, t)), float(np.polyval(py, t)), half)
            else:
                n_drop += len(run)
                print("  beads: %s f%d-%d (%d frames) reaches no contact - "
                      "clipped corona, not beads"
                      % (frames[run[0]]["file"], frames[run[0]]["index"],
                         frames[run[-1]]["index"], len(run)))
            run = []
        if j is not None:
            run.append(j)
    if bead_seen:
        print("  beads: clipped limb on %d totality frames; %d too wide or too "
              "small, %d unconnected -> %d panels in %d run(s)"
              % (bead_seen, bead_wide, n_drop, len(bead_of), n_runs))

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
            # Beads first: while one is burning it is the subject of the frame,
            # and a prominence panel is competing for a corner with something the
            # viewer can see unaided in the main view.
            bead = bead_of.get((f["file"], f["index"]))
            if bead:
                bx, by, bhalf = bead
                feats.append((bx, by, "baily's beads", args.panel/(4.0*bhalf)))
            # No prominence panels through the resolve. They are detected per
            # exposure level on a frame from that level, and every resolve level
            # is clipped over a third of the limb - so the detection runs on glare
            # and the panels magnify a washed-out patch of it. The prominences are
            # not visible to the viewer there either, which is the honest reason:
            # they emerge as the photosphere goes, and the panels start when they
            # do. The bead panel is exempt because it is pointed at the glare
            # deliberately.
            if not f.get("resolve"):
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
    for key in order:
        rows = groups[key]
        n = len(rows[0][1])
        labels = list(key[1])
        med = []
        for i in range(n):
            xs = sorted(t[1][i][0] - (t[0]["cx"] - ow/2.0) for t in rows)
            ys = sorted(t[1][i][1] - (t[0]["cy"] - oh/2.0) for t in rows)
            med.append((xs[len(xs)//2], ys[len(ys)//2]))

        """
        Assign by ANGLE around the disc, not by nearest free slot.

        Two things were wrong with nearest-free-slot plus inheritance. It packed:
        four prominences on the left of the disc took the three left slots and
        stacked their panels down one edge while the rest of the frame sat empty.
        And it tangled: a slot held while its subject moved stopped matching that
        subject's position, so 294 frames - 12% of the video - had leaders
        crossing each other, which a swap pass then had to repair. Repairing a
        symptom is not the same as removing the cause.

        The features and the slots are both rings about the same centre: the
        features on the limb, the slots around the frame. Walk both rings in the
        same direction and match them in order, and no two leaders can cross -
        by construction, with nothing to untangle afterwards. That leaves only
        WHICH slots to use, and picking the most evenly spread subset is what
        pushes the panels out around the whole frame instead of bunching them
        where the subjects happen to be.

        Cost is total leader length, less a reward for the tightest angular gap
        between the chosen slots. Length alone always bunches - the shortest
        leaders are the ones that never leave the crowded side.
        """
        cx0, cy0 = ow/2.0, oh/2.0
        f_ang = [math.atan2(p[1] - cy0, p[0] - cx0) for p in med]
        f_ord = sorted(range(n), key=lambda i: f_ang[i])

        # Slot centres, ordered around the frame the same way.
        s_ang = [math.atan2(c[1] - cy0, c[0] - cx0) for c in corner_xy]
        s_ord = sorted(range(len(corner_xy)), key=lambda c: s_ang[c])

        def gap_score(chosen):
            """Smallest angular gap between the chosen slots, in radians."""
            if len(chosen) < 2:
                return math.pi
            a = sorted(s_ang[c] for c in chosen)
            gaps = [a[i + 1] - a[i] for i in range(len(a) - 1)]
            gaps.append(a[0] + 2*math.pi - a[-1])
            return min(gaps)

        best, best_cost = None, float("inf")
        for combo in itertools.combinations(range(len(s_ord)), n):
            ring = [s_ord[c] for c in combo]          # already in cyclic order
            for rot in range(n):
                cand = [None]*n
                ok = True
                total = 0.0
                for j, fi in enumerate(f_ord):
                    slot = ring[(j + rot) % n]
                    cand[fi] = slot
                    total += leader_cost(med[fi][0], med[fi][1],
                                         corner_xy[slot], r_moon)
                    if total >= best_cost + SPREAD_WEIGHT*math.pi:
                        ok = False
                        break
                if not ok:
                    continue
                cost = total - SPREAD_WEIGHT*gap_score(ring)
                if cost < best_cost:
                    best, best_cost = cand, cost
        assigned = best

        layout[key] = tuple(assigned)

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
    # Radius the renderer occludes leader lines behind, in superpixel px.
    cfg["discR"] = r_moon
    json.dump(cfg, open(cfgpath, "w"))
    print("panel %d px at %.1fx (source box %.0f superpixel); cusps on %d of %d "
          "frames; %d clip layouts"
          % (args.panel, args.zoom, 2 * halfsp, n_cusp, len(frames), len(layout)))
    print("panels per frame: %s"
          % ", ".join("%d on %d frames" % (k, counts[k]) for k in sorted(counts)))
    print("updated %s" % cfgpath)


if __name__ == "__main__":
    main()
