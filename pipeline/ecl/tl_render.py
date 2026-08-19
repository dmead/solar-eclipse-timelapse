"""Render timelapse frames — port of tl-frames.js.

Demosaic, drizzle each group onto a 2x grid, hold the Sun still, apply the
per-frame gain chosen by `gen_timelapse.py`, draw the zoom panels, write 8-bit
PNGs for ffmpeg.

Demosaic is 2x2 superpixel, which lands exactly on 1920x1080 from this sensor and
costs no interpolation. Every frame is then rendered on a 2x grid — the sensor's
own sampling, not the superpixel grid the disc fits are measured on. For a frame
carrying a `stack` count that grid is filled by drizzle: N consecutive raw frames,
phase-correlated against the first and added onto the fine grid, so the sub-pixel
dither the mount supplied recovers detail past the native sampling.

That matters for the prominences specifically. They are Halpha, so they live
almost entirely in R, and R is only a quarter of the CFA sites — sampled at
3.43 arcsec/px, the worst-sampled signal in the capture. Drizzle is the only thing
that recovers it; a better demosaic cannot, because R is not interpolated away,
it is genuinely that sparse.

PixInsight supplied two things here and lunation supplies both: the FFT
(`core.fftreg.PhaseCorrelator` for `FFTTranslation`) and the bicubic
resample/translate that IS the drizzle kernel (`core.warp`). Note that PI's
BicubicSpline is not cv2's Keys a=-0.75 kernel, so drizzled output will not match
the PJSR render pixel for pixel — see `warp.py:27-28`. Everything else is
arithmetic that was always in plain JS.

WORKER COUNT: scale it to the physical cores. The cliff that used to be here is
gone, and it was never really about the worker count.

The old default was 4, from a 2026-08-15 measurement in which 12 workers ran
twenty times SLOWER than one. That was two faults at once and both are fixed: the
captures were on S:, a spinning disc whose readahead concurrent streams shredded,
and `scipy.fft` was unpinned, so every worker fanned its FFT across all 32 cores
— 4 workers meant 128 threads on 32 cores. With the source staged on Z: and the
pinning in `_init_worker`, re-measured 2026-08-18 on an i9-14900K (8 P-cores +
16 E-cores, 24 physical, 192 GB):

    workers      8      12      16      20      24      32
    s/frame  1.235   1.029   0.891   0.820   0.780   0.749
    vs 8      1.00    1.20    1.39    1.51    1.58    1.65

Monotonic throughout — no collapse anywhere. That run was fully cached, so it
measured CPU only; the I/O-heavy case was checked separately on the corona dwell,
291 distinct raw frames with nothing re-read, halves swapped between the arms:

    8 workers  1.305 s/frame     24 workers  0.796 s/frame     1.64x

and cold-against-cold alone is 1.71x, so more workers help on both axes here.
Returns flatten past 24, the physical core count — past it only SMT siblings are
left, and 32 buys 4% for a third more processes.

So the default is the physical core count rather than a constant; on this box
that is 24. RE-MEASURE ON A SPINNING DISC. Nothing here rules out the original
collapse returning on hardware whose readahead cannot take 24 streams.
"""

import argparse
import json
import math
import os
import time

import numpy as np
from .vendor.core.fftreg import PhaseCorrelator
from .vendor.core.warp import resample, translate

from . import affinity, font5x7, serio
from .imgio import write_png
from .source import open_source

# Drizzle scale. Everything is rendered at 2x so the drizzled totality can sit in
# the same sequence as the resampled partials with one geometry throughout.
DRIZZLE = 2
INTERP = "bicubic"

# A group spans 0.86 s, over which the measured pointing moves under 1 px. A shift
# far beyond that is a failed correlation, not motion, and the frame is left out
# rather than smeared into the stack.
MAX_GROUP_SHIFT_PX = 4.0

# Display gamma applied after the linear gain.
GAMMA = 0.65

# Fraction of brightest pixels whose centroid defines the Sun's position, used
# only when the config carries no smoothed centre for the frame.
CENTROID_FRACTION = 0.02

# Highlight shoulder. The gain maps each segment's p99 to a fixed target, but p99
# is not the peak: on a nearly full disc the photosphere covers enough of the
# frame that p99 sits INSIDE it and the brighter centre hard-clips. Measured in
# the render, 13% of the first frame was pinned at 254+ while the source frames
# clip nothing — the flat white disc was manufactured in this stage.
SHOULDER_KNEE = 0.60
# Ceiling the shoulder approaches, deliberately below 1.0: asymptoting TO white
# still renders 255 for anything far enough over the knee. At 0.965 the brightest
# possible pixel lands at 249 after the display gamma.
SHOULDER_CEIL = 0.965

# Frame gap inside one capture that counts as a discontinuity.
GAP_FRAMES = 60

# Gain ratio between consecutive video frames that counts as a cut. Set above the
# largest step the exposure ladder makes on its own inside a run (2.45x) so only
# a genuine handover between normalizations triggers it.
GAIN_JUMP = 3.0

# Percentile of the frame taken as the sky pedestal, subtracted before the gain,
# and how much of it to remove. Set the fraction to 0 to render exactly the way
# the PJSR original did.
#
# Swept on a totality and a partial frame, 2026-08-17, scoring sky black point
# against the corona still present at 2 R:
#
#   pct  frac   sky p1   corona@2R   ratio
#   5.0  0.00   0.0431      0.1529     3.5   (off — the original)
#   5.0  0.50   0.0353      0.1451     4.1
#   1.0  0.75   0.0314      0.1451     4.6
#   1.0  1.00   0.0275      0.1412     5.1   <- chosen
#
# The 1st percentile beats the 5th: on a totality frame the corona fills enough
# of the field that the 5th percentile sits inside real signal, so it removes
# corona along with sky. At the 1st, full removal blackens the sky by 36% for an
# 8% cost in the outer skirt.
PEDESTAL_PCT = 1.0
PEDESTAL_FRAC = 1.0

# Correlator for the intra-group alignment. "ported" is the literal fftalign.jsh
# port; "skimage" is an upsampled-DFT estimator accurate to ~0.01 px against the
# ported engine's 0.75 px tolerance.
#
# "ported" wins here on both counts, measured 2026-08-15 over three frames:
# 12.1 s against 21.5 s, and marginally closer to the PJSR render (52.2% of
# pixels identical against 49.8%). skimage's accuracy comes from an upsampled-DFT
# refinement that costs more than the padded square FFT it avoids, and this is
# called 19 times per output frame rather than once. `tl_track` makes the
# opposite choice for the opposite reason: there the correlation runs once per
# frame and the quantity being measured is a 0.74 px RMS residual, which the
# ported engine's tolerance cannot resolve.
DEFAULT_ENGINE = "ported"


def tune(out_dir, log=None):
    """Resolve the render constants from the config against the survey."""
    global DRIZZLE, MAX_GROUP_SHIFT_PX, GAMMA, SHOULDER_KNEE, SHOULDER_CEIL
    global PEDESTAL_PCT, PEDESTAL_FRAC, GAIN_JUMP, GAP_FRAMES, PANEL_EXPOSE
    global PANEL_EXPOSE_PCT, PANEL_EXPOSE_TARGET, DEFAULT_ENGINE
    from .params import load

    P = load(out_dir, create=False)
    DRIZZLE = int(P.get("render.drizzle") or DRIZZLE)
    MAX_GROUP_SHIFT_PX = P.px("render.max_group_shift_r")
    GAMMA = P.get("render.gamma", GAMMA)
    SHOULDER_KNEE = P.get("render.shoulder_knee", SHOULDER_KNEE)
    SHOULDER_CEIL = P.get("render.shoulder_ceil", SHOULDER_CEIL)
    PEDESTAL_PCT = P.get("render.pedestal_pct", PEDESTAL_PCT)
    PEDESTAL_FRAC = P.get("render.pedestal_frac", PEDESTAL_FRAC)
    GAIN_JUMP = P.get("render.gain_jump", GAIN_JUMP)
    GAP_FRAMES = P.get("render.gap_frames", GAP_FRAMES)
    PANEL_EXPOSE = P.get("panels.expose", PANEL_EXPOSE)
    PANEL_EXPOSE_PCT = P.get("panels.expose_pct", PANEL_EXPOSE_PCT)
    PANEL_EXPOSE_TARGET = P.get("panels.expose_target", PANEL_EXPOSE_TARGET)
    DEFAULT_ENGINE = P.get("render.engine", DEFAULT_ENGINE)
    if log:
        log(f"  tuned to r={P.radius_px:.0f}px: drizzle x{DRIZZLE}, "
            f"group shift <= {MAX_GROUP_SHIFT_PX:.1f} px")
    return P


def default_workers():
    """Physical cores, which is where the throughput curve flattens.

    Counted rather than hardcoded so this does not oversubscribe a smaller
    machine, and falling back to the logical count wherever the topology query
    does not work.
    """
    return len(affinity.core_groups()) or (os.cpu_count() or 4)


__all__ = ["render_frames", "crop_gain", "accumulate", "centroid", "DEFAULT_ENGINE"]


# ------------------------------------------------------------------- drizzle

def accumulate(plane, tx=0.0, ty=0.0, acc=None):
    """Place one plane onto the fine grid, shifted by (tx, ty) plane pixels.

    With translation-only alignment this is drizzle at pixfrac 1: upsample by the
    scale factor, offset by scale*shift, add. The upsample is bicubic rather than
    nearest because the dither here is a smooth drift, not a random walk — 20
    frames land in only 8 or 9 of the 16 sub-pixel cells, and nearest-neighbour on
    that leaves visible 2x2 stair-stepping in the cells nothing reached.
    """
    up = resample(plane, DRIZZLE, INTERP)
    if tx or ty:
        up = translate(up, DRIZZLE * tx, DRIZZLE * ty, interp=INTERP)
    if acc is None:
        return up
    acc += up
    return acc


def centroid(g):
    """Intensity-weighted centroid of the brightest pixels.

    The partial crescent, or the corona once the filter is off — both are centred
    on the Sun. Only a fallback: the smoothed disc track is used when present,
    because per-frame detection jitters by a pixel or two and gives up entirely on
    the thinnest crescents, either of which shows up as the picture twitching.
    """
    h, w = g.shape
    lo, hi = float(g.min()), float(g.max())
    bins = 1024
    scale = (bins - 1) / (hi - lo) if hi > lo else 0.0
    idx = np.rint((g - lo) * scale).astype(np.int32)
    hist = np.bincount(idx.reshape(-1), minlength=bins)

    want = max(1, int(round(CENTROID_FRACTION * g.size)))
    acc, b = 0, bins - 1
    while b > 0 and acc < want:
        acc += int(hist[b])
        b -= 1
    thr = lo + b / (scale or 1.0)

    d = g - thr
    np.maximum(d, 0, out=d)
    sw = float(d.sum())
    if sw <= 0:
        return w / 2.0, h / 2.0
    ys, xs = np.arange(h), np.arange(w)
    return float((d.sum(0) * xs).sum() / sw), float((d.sum(1) * ys).sum() / sw)


def apply_gain(v, gain, pedestal=0.0):
    """Sky pedestal, then linear gain, then the highlight shoulder, in place.

    Shared by the main view and the inset panels so they cannot drift apart.
    The panels used to skip this entirely — `draw_insets` samples the fine grid
    for geometric reasons and that also bypassed the photometry, so a panel
    showed raw linear data inside a frame multiplied by the segment gain. At
    1.7x that reads as slightly off; at the 27x the long exposures use it reads
    as a different picture.

    The PEDESTAL is why the panels looked better. Totality sky is dusk, not
    black, and that offset is additive — multiplying it by a 27x segment gain
    turns a dark sky into a grey plate and takes the lunar disc with it. The
    ungained panels never had that problem, which is what made their blacks
    look right. Subtracting the pedestal before the gain gives the whole frame
    the panels' contrast while keeping the per-segment normalisation that stops
    the video flickering at exposure changes. `corona-stretch` removes the same
    pedestal for the same reason, and in the same order.
    """
    if pedestal:
        v -= np.float32(pedestal)
    v *= np.float32(gain)
    span = SHOULDER_CEIL - SHOULDER_KNEE
    over = v > SHOULDER_KNEE
    if over.any():
        v[over] = SHOULDER_KNEE + span * np.tanh((v[over] - SHOULDER_KNEE) / span)
    np.maximum(v, 0, out=v)
    return v


def sky_pedestal(plane):
    """The additive sky level of one plane, as a low percentile."""
    if not PEDESTAL_FRAC:
        return 0.0
    return float(np.percentile(plane, PEDESTAL_PCT)) * PEDESTAL_FRAC


# A panel is exposed for ITS OWN subject, never brighter than the frame gain.
#
# The frame gain normalizes the corona, and a panel inheriting it shows whatever
# it is pointed at multiplied by a number chosen for something else. Around the
# contacts that number reaches 27x while the panel is looking at photosphere, and
# every panel in the frame renders as a flat white square.
#
# Pulling the gain down only ever helps where the sensor was not already clipped:
# a saturated plateau goes from a white disc to a grey one and gains no detail,
# which is why gen_insets also gates the bead panel on the arc being thin enough
# to have structure left in it. The two work together and neither is sufficient.
PANEL_EXPOSE = True
PANEL_EXPOSE_PCT = 99.5
PANEL_EXPOSE_TARGET = 0.85


def panel_gain(samples, gain, pedestals=None):
    """Gain for one panel: the frame's, reduced until the subject fits.

    Measured across all three channels at once, so a panel is not white-balanced
    by accident — the prominences are almost pure R and scaling that channel on
    its own would drain the colour out of exactly the feature the panel exists to
    show.
    """
    if not PANEL_EXPOSE:
        return gain
    hi = 0.0
    for c, s in enumerate(samples):
        ped = pedestals[c] if pedestals else 0.0
        hi = max(hi, float(np.percentile(s, PANEL_EXPOSE_PCT)) - ped)
    if hi <= 0:
        return gain
    return min(gain, PANEL_EXPOSE_TARGET / hi)


def crop_gain(src, ox, oy, out_w, out_h, gain, pedestal=0.0):
    """Bilinear sub-pixel crop, linear gain and highlight shoulder in one pass.

    Outside the sensor this leaves black. Clamping to the edge pixel would smear a
    bright streak along the border and pretend it was data.
    """
    h, w = src.shape
    dst = np.zeros((out_h, out_w), dtype=np.float32)

    sy = oy + np.arange(out_h)
    sx = ox + np.arange(out_w)
    y0 = np.floor(sy).astype(np.int64)
    x0 = np.floor(sx).astype(np.int64)
    fy = (sy - y0).astype(np.float32)
    fx = (sx - x0).astype(np.float32)

    vy = (y0 >= 0) & (y0 + 1 < h)
    vx = (x0 >= 0) & (x0 + 1 < w)
    if not vy.any() or not vx.any():
        return dst

    yy, xx = y0[vy], x0[vx]
    fyv, fxv = fy[vy][:, None], fx[vx][None, :]
    a00 = src[np.ix_(yy, xx)]
    a01 = src[np.ix_(yy, xx + 1)]
    a10 = src[np.ix_(yy + 1, xx)]
    a11 = src[np.ix_(yy + 1, xx + 1)]
    a = a00 + (a01 - a00) * fxv
    bq = a10 + (a11 - a10) * fxv
    v = apply_gain(a + (bq - a) * fyv, gain, pedestal)

    dst[np.ix_(np.nonzero(vy)[0], np.nonzero(vx)[0])] = v
    return dst


# --------------------------------------------------------------------- panels

class _Canvas:
    """The three output planes plus the primitives that draw on them."""

    EDGE = 0.92          # line and border brightness

    def __init__(self, planes):
        self.planes = planes
        self.h, self.w = planes[0].shape

    def set_px(self, x, y, v):
        x, y = int(x), int(y)
        if x < 0 or y < 0 or x >= self.w or y >= self.h:
            return
        for p in self.planes:
            p[y, x] = v

    def line(self, x0, y0, x1, y1, occlude=None):
        """Draw a line, optionally skipping the part inside a circle.

        `occlude` is (cx, cy, r). A leader whose subject is on the far side of
        the disc has to be drawn across it, and there is no way to choose corners
        out of that: measured over the whole cut, every frame with two or more
        panels has NO crossing-free assignment, because the features sit ON the
        limb and the corners are outside it. The worst leader ran 1123 px over a
        disc 1168 px across - the full diameter.

        Skipping the covered span makes the line read as passing behind the Moon,
        which is where it is. That is also true of the picture: the disc is the
        nearest object in the frame, so a hairline crossing it is the only part of
        this annotation that contradicts the scene. It is the darkest region too,
        which is why a line there is far more visible than the same line over
        corona.
        """
        n = int(math.ceil(max(abs(x1 - x0), abs(y1 - y0))))
        if n <= 0:
            self.set_px(round(x0), round(y0), self.EDGE)
            self.set_px(round(x0) + 1, round(y0), self.EDGE)
            return
        ocx = ocy = orr = None
        if occlude:
            ocx, ocy, orr = occlude
            orr *= orr
        for s in range(n + 1):
            fx = x0 + (x1 - x0) * s / n
            fy = y0 + (y1 - y0) * s / n
            if orr is not None:
                dx, dy = fx - ocx, fy - ocy
                if dx * dx + dy * dy < orr:
                    continue
            x, y = math.floor(fx + 0.5), math.floor(fy + 0.5)
            self.set_px(x, y, self.EDGE)
            self.set_px(x + 1, y, self.EDGE)

    def rect(self, x0, y0, x1, y1):
        self.line(x0, y0, x1, y0)
        self.line(x1, y0, x1, y1)
        self.line(x1, y1, x0, y1)
        self.line(x0, y1, x0, y0)

    def text(self, s, x, y, z):
        """Label on a darkened plate.

        The plate MULTIPLIES rather than fills, which keeps it invisible against
        the black sky where a solid box would be a grey rectangle, while still
        carrying white text over the photosphere.
        """
        s = str(s).upper()
        tw, th, pad = font5x7.text_width(s, z), 7 * z, 2 * z
        x, y = int(x), int(y)

        y0, y1 = max(0, y - pad), min(self.h, y + th + pad)
        x0, x1 = max(0, x - pad), min(self.w, x + tw + pad)
        if y1 > y0 and x1 > x0:
            for p in self.planes:
                p[y0:y1, x0:x1] *= 0.25

        mask = font5x7.glyph_mask(s, z)
        mh, mw = mask.shape
        gy0, gy1 = max(0, y), min(self.h, y + mh)
        gx0, gx1 = max(0, x), min(self.w, x + mw)
        if gy1 <= gy0 or gx1 <= gx0:
            return
        sub = mask[gy0 - y:gy1 - y, gx0 - x:gx1 - x]
        for p in self.planes:
            p[gy0:gy1, gx0:gx1][sub] = self.EDGE


def draw_insets(out_planes, fine, ox2, oy2, insets, panel, zoom,
                gain=1.0, pedestals=None, disc_r=None):
    """Zoomed panels in the corners, with a box and leader lines to the source.

    The panels are sampled from the FINE grid rather than from the cropped output,
    so a box near the edge of the crop still gets its full surroundings, and the
    magnification is honest. Sampling is bilinear — nearest would show the drizzle
    grid as stair-stepping at this magnification, which reads as detail that is
    not there.

    There is NOT always one panel per corner. Each inset names its own corner and
    carries the name of the thing it follows, and `gen_insets.py` emits only the
    features that exist in that frame: before first contact there are no cusps,
    the sunspot spends part of the eclipse behind the Moon, and a totality level
    may show fewer than four prominences worth pointing at.
    """
    M = 24                                   # margin from the frame edge
    cv = _Canvas(out_planes)
    out_h2, out_w2 = cv.h, cv.w
    fh, fw = fine[0].shape
    # Up to eight slots, in the order gen_insets assigns them: the four corners,
    # then left, right, top, bottom. The last two exist only where the panel was
    # sized to clear the disc; gen_insets decides that and simply never emits a
    # corner index for them otherwise. See `corner_xy` there.
    cx_mid, cy_mid = (out_w2 - panel) // 2, (out_h2 - panel) // 2
    corners = [(M, M), (M, out_h2 - panel - M),
               (out_w2 - panel - M, M),
               (out_w2 - panel - M, out_h2 - panel - M),
               (M, cy_mid), (out_w2 - panel - M, cy_mid),
               (cx_mid, M), (cx_mid, out_h2 - panel - M)]

    for k, ins in enumerate(insets[:4]):
        # Source centre: superpixel coordinates, same units as the disc track.
        scx, scy = DRIZZLE * ins["cx"], DRIZZLE * ins["cy"]
        px, py = corners[ins.get("corner", k)]
        # Each inset may magnify at its own rate: a cusp is a needle and needs a
        # much tighter box than a sunspot group or a prominence.
        z = ins.get("zoom") or zoom
        hb = panel / z / 2                   # half the source box, fine px

        u = np.arange(panel)
        sxs, sys = scx - hb + u / z, scy - hb + u / z
        x0 = np.floor(sxs).astype(np.int64)
        y0 = np.floor(sys).astype(np.int64)
        fx = (sxs - x0).astype(np.float32)[None, :]
        fy = (sys - y0).astype(np.float32)[:, None]
        okx = (x0 >= 0) & (x0 + 1 < fw)
        oky = (y0 >= 0) & (y0 + 1 < fh)

        xc, yc = np.clip(x0, 0, fw - 2), np.clip(y0, 0, fh - 2)
        keep = oky[:, None] & okx[None, :]
        # Sample all three channels BEFORE gaining any of them: the exposure is a
        # property of the panel as a whole and has to be measured across the set.
        samples = []
        for p in fine:
            a00 = p[np.ix_(yc, xc)]
            a01 = p[np.ix_(yc, xc + 1)]
            a10 = p[np.ix_(yc + 1, xc)]
            a11 = p[np.ix_(yc + 1, xc + 1)]
            a = a00 + (a01 - a00) * fx
            b = a10 + (a11 - a10) * fx
            samples.append(np.where(keep, a + (b - a) * fy, 0.0).astype(np.float32))

        # An inset may name its own exposure: a number scales the frame gain, and
        # "auto" fits the gain to the panel's own content.
        ex = ins.get("expose")
        if ex == "auto" or (ex is None and PANEL_EXPOSE):
            g = panel_gain(samples, gain, pedestals)
        elif ex is not None:
            g = gain * float(ex)
        else:
            g = gain

        for c, v in enumerate(samples):
            ped = pedestals[c] if pedestals else 0.0
            out_planes[c][py:py + panel, px:px + panel] = apply_gain(v, g, ped)

        # The source box, in output coordinates.
        bx0, by0 = scx - hb - ox2, scy - hb - oy2
        bx1, by1 = scx + hb - ox2, scy + hb - oy2
        cv.rect(bx0, by0, bx1, by1)
        cv.rect(px, py, px + panel - 1, py + panel - 1)

        # Leader lines: join the two box corners on the side FACING the panel to
        # the two panel corners facing the box. Choosing the near side per panel
        # rather than a fixed pair is what stops the lines crossing the frame
        # diagonally when a box drifts past the panel it belongs to — which the
        # cusps do, since they travel right across the disc.
        panel_left = (px + panel / 2) < (bx0 + bx1) / 2
        sbx = bx0 if panel_left else bx1
        spx = px + panel - 1 if panel_left else px
        # The framed disc is always dead centre of the output: the crop is
        # built around the same cx/cy the disc track carries.
        occ = (out_w2 / 2, out_h2 / 2, DRIZZLE * disc_r) if disc_r else None
        cv.line(sbx, by0, spx, py, occlude=occ)
        cv.line(sbx, by1, spx, py + panel - 1, occlude=occ)

        # Name the subject, OUTSIDE the panel: below a top panel and above a
        # bottom one, so it never covers the magnified view.
        label = ins.get("label")
        if label:
            top = py < out_h2 / 2
            z_t = font5x7.text_scale(str(label).upper(), panel)
            tw = font5x7.text_width(str(label).upper(), z_t)
            ty = py + panel + 3 * z_t if top else py - 7 * z_t - 3 * z_t
            tx = px if (px + panel / 2 < out_w2 / 2) else px + panel - tw
            cv.text(label, tx, ty, z_t)


# ---------------------------------------------------------------- the render

def render_frames(frames, cfg, engine=None, log=print):
    """Render `frames` (a slice of timelapse.json's frames) to 8-bit PNGs."""
    t0 = time.time()
    engine = engine or cfg.get("correlator", DEFAULT_ENGINE)
    out_dir = cfg["outDir"]
    os.makedirs(out_dir, exist_ok=True)

    out_w, out_h = cfg.get("outW", 1280), cfg.get("outH", 720)
    out_w2, out_h2 = DRIZZLE * out_w, DRIZZLE * out_h
    dissolve = cfg.get("dissolve", 3)
    panel, zoom = cfg.get("insetPanel", 420), cfg.get("insetZoom", 3)
    log(f"timelapse: {len(frames)} frames -> {out_dir}")
    log(f"  output {out_w2}x{out_h2} (drizzle x{DRIZZLE}), dissolve {dissolve} "
        f"frames, correlator {engine}")

    ser = None
    cur_path = ""
    dissolve_file, dissolve_left, dissolve_src, prev_out = None, 0, None, None
    dissolve_index = -1
    dissolve_gain = None
    written = 0

    try:
        for k, fr in enumerate(frames):
            if fr["src"] != cur_path:
                if ser is not None:
                    ser.close()
                ser = open_source(fr["src"])
                cur_path = fr["src"]
                log(f"  open {fr['file']} ({ser.raw_width >> 1}x{ser.raw_height >> 1})")

            R, G, B = ser.planes(fr["index"])

            # Alignment is measured once, on G, and applied to all three channels.
            # Per-channel alignment would let the colours drift apart by a
            # fraction of a pixel, and G carries the most signal anyway.
            #
            # The reference is the group's FIRST frame, which is also the frame
            # the disc track was measured on — so the stacked result sits exactly
            # where the unstacked one would have, and stacked and unstacked frames
            # can be mixed in one sequence without the subject stepping.
            stack = int(fr.get("stack", 1) or 1)
            if stack > 1:
                aligner = PhaseCorrelator(use_gradient=False, engine=engine)
                aligner.initialize(G)
                acc = [accumulate(R), accumulate(G), accumulate(B)]
                n_stack = 1
                for j in range(1, stack):
                    idx = fr["index"] + j
                    if idx >= ser.frame_count:
                        break
                    R2, G2, B2 = ser.planes(idx)
                    sh = aligner.evaluate(G2)
                    dx, dy = ((sh["dx"], sh["dy"]) if isinstance(sh, dict) else sh)
                    if abs(dx) > MAX_GROUP_SHIFT_PX or abs(dy) > MAX_GROUP_SHIFT_PX:
                        continue
                    for c, plane in enumerate((R2, G2, B2)):
                        accumulate(plane, -dx, -dy, acc[c])
                    n_stack += 1
                fine = [a / n_stack for a in acc]
            else:
                fine = [accumulate(R), accumulate(G), accumulate(B)]

            # Hold the Sun still. The centre comes from the smoothed disc track,
            # not from this frame. The window is NOT clamped inside the sensor:
            # clamping would keep the frame full at the cost of letting the Sun
            # slide, which is the whole problem, and it also crops away corona the
            # sensor did record. Where the window reaches past the edge the
            # renderer pads black — the honest statement that nothing was there.
            if fr.get("cx") is not None:
                ox, oy = fr["cx"] - out_w / 2, fr["cy"] - out_h / 2
            else:
                ccx, ccy = centroid(G)
                ox = min(max(ccx - out_w / 2, 0), G.shape[1] - out_w)
                oy = min(max(ccy - out_h / 2, 0), G.shape[0] - out_h)

            ox2, oy2 = DRIZZLE * ox, DRIZZLE * oy
            # Measured per channel on the drizzled plane, so the panels and the
            # main view subtract the same number.
            peds = [sky_pedestal(f) for f in fine]
            out_planes = [crop_gain(f, ox2, oy2, out_w2, out_h2, fr["gain"], p)
                          for f, p in zip(fine, peds)]

            # Cross-dissolve across a discontinuity. Within a capture successive
            # video frames are 0.86 s apart and the Moon advances 0.24 px. At a
            # boundary the recorder was flushing for 25 to 490 s, so the Moon
            # jumps 7 to 136 px in a single frame and it reads as a jolt. Nothing
            # can fill that gap; a short dissolve just stops the discontinuity
            # landing on one frame. Any real-time gap counts, not just a change of
            # file: dropping blown frames leaves holes of 8 to 13 s inside a
            # capture, across which the Sun has moved just as much.
            # A big GAIN step counts as a discontinuity too, even between two
            # consecutive raw frames. Where the second-contact resolve hands over
            # to the corona exposure the gain goes 4.19 to 27.31 across f1169 to
            # f1170 - 43 ms apart, so the sky is identical and only the camera
            # changed - and the rendered frame jumps 2.3x in mean. Normalization
            # is meant to remove exactly that, and cannot here: the resolve is
            # held down by the transition ceiling because its highlights are
            # clipped. One frame of it in a fast sequence went unnoticed; arriving
            # at the end of eight seconds of slow motion it reads as a flash.
            g_prev = dissolve_gain or fr["gain"]
            jumped = (fr["file"] != dissolve_file
                      or fr["index"] - dissolve_index > GAP_FRAMES
                      or max(fr["gain"], g_prev) / max(min(fr["gain"], g_prev), 1e-6)
                      > GAIN_JUMP)
            dissolve_gain = fr["gain"]
            if jumped:
                if dissolve_file is not None and prev_out is not None and dissolve > 0:
                    dissolve_left, dissolve_src = dissolve, prev_out
                dissolve_file = fr["file"]
            dissolve_index = fr["index"]
            if dissolve_left > 0 and dissolve_src is not None:
                w_old = dissolve_left / (dissolve + 1)
                for c in range(3):
                    out_planes[c] *= (1 - w_old)
                    out_planes[c] += dissolve_src[c] * w_old
                dissolve_left -= 1

            # Remember the main view WITHOUT the panels, then draw them.
            # Cross-fading the panels turns their outlines and leader lines into
            # doubled ghost lines for the length of the dissolve. The main view
            # should dissolve; the annotation on top of it should not.
            prev_out = [p.copy() for p in out_planes]

            # The panels are drawn at FULL strength on every frame, always. Both
            # cleverer schemes tried across a dissolve were worse: sliding a box
            # between its old and new position samples the PANEL there too, so
            # every zoom showed the wrong patch of sky while it drifted; fading
            # the annotation on the dissolve weight reads as the corners flashing.
            if fr.get("insets"):
                draw_insets(out_planes, fine, ox2, oy2, fr["insets"], panel,
                            zoom, gain=fr["gain"], pedestals=peds,
                            disc_r=cfg.get("discR"))

            rgb = np.stack(out_planes, axis=-1)
            np.clip(rgb, 0.0, 1.0, out=rgb)
            rgb **= np.float32(GAMMA)

            # Number by the frame's own sequence when the driver supplied one:
            # under parallel sharding a frame's position within its shard is not
            # its position in the video.
            seq = fr.get("seq", k)
            write_png(f"{out_dir}/seq_{seq:05d}.png", rgb, bit_depth=8)
            written += 1

            if (k + 1) % 100 == 0:
                log(f"  rendered {k + 1}/{len(frames)} "
                    f"({(time.time() - t0) / (k + 1):.2f} s/frame)")
    finally:
        if ser is not None:
            ser.close()

    log(f"  {written} frames in {(time.time() - t0) / 60:.1f} min")
    return written


# Thread-limit variables the numeric libraries read AT IMPORT. They have to be
# set in the PARENT before the pool spawns, because a spawned child imports numpy
# and its BLAS afresh and reads them then — setting them inside the worker
# function is too late and does nothing, which is the bug this replaces.
_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _init_worker(cpus=None, slot=None):
    """One thread per worker process, and optionally one core to run it on.

    Three separate thread pools have to be pinned, and missing any one of them
    oversubscribes the box:

    - OpenCV, via `setNumThreads`, which does work at runtime.
    - The BLAS behind numpy, via the environment above, which the parent sets.
    - **scipy.fft**, which `PhaseCorrelator` calls with `workers=-1` and which is
      55% of the per-frame cost. This is the one that was missed. Four workers
      each fanning the FFT across 32 cores is 128 threads on 32 cores, and it is
      the most likely reason more workers measured SLOWER than one.
    """
    import cv2

    cv2.setNumThreads(1)

    # Claim a core. The pool gives every worker the same initargs, so the slot is
    # taken from a shared counter rather than passed in - there is no worker index
    # to key on, and two workers on one core is worse than none pinned at all.
    if cpus and slot is not None:
        with slot.get_lock():
            i = slot.value
            slot.value += 1
        if i < len(cpus):
            affinity.pin(cpus[i])


def _shard(args):
    cfg, a, b, engine = args
    import scipy.fft

    # set_workers is a context manager, so the limit has to wrap the work
    # rather than be set once at import.
    with scipy.fft.set_workers(1):
        return render_frames(cfg["frames"][a:b], cfg, engine=engine,
                             log=lambda m: None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="S:/solar-eclipse/out/configs/timelapse.json")
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help="contiguous shards rendered in parallel processes. "
                         "Defaults to the physical core count, where the "
                         "measured curve flattens; read the module docstring "
                         "before changing it on other storage.")
    ap.add_argument("--start", type=int, default=0,
                    help="first frame index in the config to render")
    ap.add_argument("--limit", type=int, default=None,
                    help="render at most N frames from --start")
    ap.add_argument("--out-dir", default=None, help="override cfg.outDir")
    ap.add_argument("--data-dir", default=None,
                    help="read the captures from here instead of the directory "
                         "baked into the config's absolute frames[].src paths, "
                         "so the source can be staged on a faster volume "
                         "without rewriting timelapse.json")
    ap.add_argument("--engine", default=None, choices=["skimage", "ported"],
                    help="intra-group correlator (default skimage)")
    ap.add_argument("--resume", action="store_true",
                    help="skip frames already written as complete PNGs, so a "
                         "killed render continues instead of starting over")
    ap.add_argument("--affinity", action="store_true",
                    help="pin each worker to its own performance core (hybrid "
                         "CPUs; ignored when workers exceed the P-core count)")
    args = ap.parse_args(argv)

    # Resolve the scale-dependent constants before anything reads them. The
    # config lives beside the timelapse config, one level up from configs/.
    out_root = os.path.dirname(os.path.dirname(os.path.abspath(args.config)))
    try:
        tune(out_root, log=print)
    except SystemExit:
        print("  no survey/config found - using built-in defaults")

    with open(args.config, encoding="utf-8-sig") as f:
        cfg = json.load(f)
    if args.out_dir:
        cfg["outDir"] = args.out_dir
    serio.restage(cfg, args.data_dir)
    # Stamp the global sequence number before any slicing, so --start/--limit
    # render the same numbered frames they would in a full run.
    for i, fr in enumerate(cfg["frames"]):
        fr.setdefault("seq", i)

    if args.resume:
        """
        Drop frames already on disk, but only COMPLETE ones.

        A render killed mid-frame leaves a truncated PNG, and skipping that would
        put a corrupt frame in the video with nothing to show it went wrong. The
        last eight bytes of a PNG are its IEND chunk, so completeness is one
        last eight bytes start with b'IEND' on any complete file, and checking
        it is one seek rather than a decode.
        """
        keep, done = [], 0
        for fr in cfg["frames"]:
            p = f"{cfg['outDir']}/seq_{fr['seq']:05d}.png"
            ok = False
            try:
                if os.path.getsize(p) > 8:
                    with open(p, "rb") as fh:
                        fh.seek(-8, os.SEEK_END)
                        ok = fh.read(8)[:4] == b"IEND"
            except OSError:
                ok = False
            if ok:
                done += 1
            else:
                keep.append(fr)
        print(f"resume: {done} frames already written, {len(keep)} to render")
        cfg["frames"] = keep
        if not keep:
            print("nothing to do")
            return
    end = args.start + args.limit if args.limit else len(cfg["frames"])
    cfg["frames"] = cfg["frames"][args.start:end]

    # `seq` was stamped above, before slicing. Without it a frame's position
    # within its shard would be taken as its position in the video and every
    # shard would overwrite the others' output — the PJSR driver assigned `seq`
    # for exactly this reason.
    n = len(cfg["frames"])
    if args.workers <= 1:
        render_frames(cfg["frames"], cfg, engine=args.engine)
        return

    # Contiguous shards: each keeps one SER open and its own dissolve state, the
    # same trade the PJSR driver made. The launch mutex it needed is gone —
    # that existed only for the PixInsight instance-slot race.
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    # Must happen before the pool is created: spawned children inherit this
    # environment and read it while importing numpy.
    for k in _THREAD_ENV:
        os.environ[k] = "1"

    step = math.ceil(n / args.workers)
    jobs = [(cfg, a, min(a + step, n), args.engine) for a in range(0, n, step)]
    print(f"rendering {n} frames in {len(jobs)} shards on {args.workers} workers")
    t0 = time.time()
    cpus, slot = [], None
    if args.affinity:
        cpus = affinity.plan(args.workers)
        if cpus:
            slot = multiprocessing.Value("i", 0)
            print(f"pinning workers to performance cores {cpus}")
        else:
            print("no distinct performance cores to pin to; leaving it to the OS")

    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(cpus, slot)) as ex:
        total = sum(ex.map(_shard, jobs))
    print(f"{total} frames in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
