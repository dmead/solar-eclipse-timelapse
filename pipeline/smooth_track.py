"""Stage F pass 2 - turn per-frame disc fits into a Sun-fixed track.

Two different jobs, and they want opposite things.

On the PARTIAL phases the disc fit is precise: measured across these captures it
sits 0.15 to 0.37 px from the mean of its two neighbours, and under 0.9 px at
p90. Real mount wander over the same captures runs 1 to 26 px. So each frame is
simply centred on its OWN fit - that cancels the shake exactly, and puts back a
third of a pixel of fit noise, which is invisible. A modelled track cannot do
this: its slew limit exists precisely to stop it chasing fast motion, so it
leaves the shake in.

Through TOTALITY the detector has no photosphere and measures the Moon's limb
from a brightness gradient. That scatters several pixels and its errors are
uncorrelated frame to frame, so following it injects more motion than it removes.
There the track is modelled - a robust line through the fits, with only the
correlated part of the residual followed.

Between captures the mount was nudged by hand - the measured jumps run to
hundreds of pixels - so a discontinuity there is expected rather than smoothed
away. Because every frame is centred on the Sun, that discontinuity never reaches
the video.

    python smooth_track.py --out S:/solar-eclipse/out

Rewrites `configs/timelapse.json` with a smoothed centre on every frame and an
output window sized to the track.
"""

import argparse
import json
import math
import os

# Quality gate for a circle fit. Arc coverage is the informative one: a fit can
# have plenty of inliers and still be badly constrained if they all lie on a
# short arc, which is exactly the thin-crescent case.
MIN_ARC_DEG = 100
MIN_INLIERS = 60
MAX_RMS_PX = 8.0

# Robust line fit: iterations and the residual cut, in units of the median
# absolute residual.
FIT_PASSES = 3
OUTLIER_K = 2.5

# A segment whose fitted drift rate exceeds this multiple of the median rate
# across all segments is treated as bad fits rather than a fast Sun, and is held
# at a constant position instead.
MAX_RATE_FACTOR = 4.0

# Candidate output sizes are searched down from the full frame in this step.
SIZE_STEP = 20
# Window is sized to hold the corona out to this many solar radii, shrinking
# only if too many frames would need padding, and never below MIN_RADII.
CORONA_RADII = 2.2
MIN_RADII = 1.6
MAX_PADDED_FRACTION = 0.35


def robust_line(ts, vs):
    """Least-squares line through (t, v) with iterative outlier rejection."""
    idx = list(range(len(ts)))
    a = b = 0.0
    for _ in range(FIT_PASSES):
        n = len(idx)
        if n < 2:
            break
        st = sum(ts[i] for i in idx)
        sv = sum(vs[i] for i in idx)
        stt = sum(ts[i] * ts[i] for i in idx)
        stv = sum(ts[i] * vs[i] for i in idx)
        den = n * stt - st * st
        if abs(den) < 1e-9:
            a, b = 0.0, sv / n
            break
        a = (n * stv - st * sv) / den
        b = (sv - a * st) / n
        res = sorted(abs(vs[i] - (a * ts[i] + b)) for i in idx)
        med = res[len(res) // 2] if res else 0.0
        cut = max(OUTLIER_K * med, 0.5)
        keep = [i for i in idx if abs(vs[i] - (a * ts[i] + b)) <= cut]
        if len(keep) < max(4, 0.3 * len(idx)):
            break
        idx = keep
    return a, b, len(idx)


# Residual larger than this is a failed fit that jumped, not wobble.
MAX_RESIDUAL_PX = 25.0

# A frame is dropped when its own detection sits this many times the capture's
# OWN typical scatter away from the settled model, and at least TRUST_FLOOR_PX.
# Adaptive because the two detectors have very different precision: the circle
# fit on a filtered disc is good to about a pixel, while the ring search on the
# corona scatters several. A single fixed tolerance either keeps a knocked scope
# or throws away most of totality - at 6 px flat it discarded 89 good totality
# frames.
TRUST_K = 6.0
TRUST_FLOOR_PX = 8.0

# If more than this fraction of a capture fails the trust check, the detector is
# at fault rather than the frames, and the capture is kept whole.
MOSTLY_FAILED = 0.5

# Detection spread above which a mostly-failed capture is treated as unplaceable
# rather than merely noisy, and dropped instead of kept.
UNPLACEABLE_SPREAD_PX = 60.0

# Smoothing widths, in samples, applied to the residual series. Wide enough that
# detection noise averages out; narrow enough to still follow wobble on a
# timescale of a few seconds, which is what the mount and seeing actually do.
MEDIAN_W = 7
MEAN_W = 9

# Lag-1 autocorrelation below which a residual series is treated as measurement
# noise and NOT followed. Real mount motion is strongly correlated frame to frame
# (filtered captures measure +0.6 to +0.75); ring-search scatter on the corona is
# not (+0.2 to +0.5 at 7 px rms, against ~1 px on a filtered disc). Following that
# put a visible wobble through totality, with the model running at its slew limit
# the whole time just to chase noise.
MIN_LAG1 = 0.55

# Per-exposure detection bias inside totality: minimum run length worth
# correcting, and the largest correction allowed.
# Residual excursion above which a capture is followed rather than modelled.
# Set from the data, which separates cleanly: 480, 142, 39, 37, 22 and 20 px for
# the captures where the scope was handled or ringing, against 4.8 px and below
# for every ordinary one. 15 sits in that gap.
FOLLOW_EXCURSION_PX = 15.0
FOLLOW_STEP_PX = 8.0
FOLLOW_MEDIAN_W = 7

# Frame-to-frame model motion above which the pointing is thrashing rather than
# settling, and the frame is cut. Stable stretches inside 14_05_08 step under
# 5 px; its thrashing steps run 50 to 370 px.
THRASH_PX = 15.0

# The one thing a per-frame follow must NOT do is follow a fit that has snapped to
# a different solution. As the crescent thins the disc fit can be confident and
# wrong: 14_11_23 reports arc 140-170 deg, 120 inliers and rms 1.7 while jumping
# 40 to 140 px per frame through its last 40%. Those frames are cut, not tracked.
#
# Departure from the capture's own robust drift line separates the two cleanly on
# this data: 0.5 to 5.9 px on the ordinary captures, 18.4 px on 14_17_51 where the
# scope was genuinely ringing after the film went back on, against 145.7 px for
# the snapped fits. 40 sits in that gap, so real handling is followed and a snap
# is cut.
FOLLOW_LINE_TOL_PX = 40.0

LEVEL_BIAS_MIN_N = 4
LEVEL_BIAS_MAX_PX = 12.0

# Fewest correlation measurements a capture needs before totality is placed from
# them rather than from the line.
CORONA_MIN_N = 8

# Half-width, in SECONDS, of the smoothing applied to the correlation track.
#
# In time rather than in samples, because totality is sampled at two very
# different rates: 0.86 s apart over most of it, 0.043 s apart through the
# resampled prominence level. A fixed tap count would barely touch the noise in
# one and erase real motion in the other; a fixed duration gives about five taps
# at the coarse cadence and sixty at the fine one, which is what each needs.
#
# The correlation's own noise is uncorrelated frame to frame - measured at 0.39 px
# inside the dense level, where the Moon moves 0.005 px between samples so nearly
# all of it is measurement - and averaging cuts it by root N. Real mount motion is
# smooth at this cadence, so a mean over a few seconds is unbiased against
# anything but its curvature, which is far smaller than the noise being removed.
CORONA_SMOOTH_S = 1.5

# Hard ceiling on how fast the tracked residual may move, px per sample. Beyond
# the shared drift the Sun cannot lurch, so this guarantees smoothness even where
# the detections are noisy - 14_19_28 was otherwise handing the model 5 px steps
# every 0.9 s, which is faster than any mount moves and reads as jitter.
MAX_RESID_STEP = 1.0


def fill_gaps(vs):
    """Replace None with the nearest available value on either side."""
    out = list(vs)
    last = None
    for i, v in enumerate(out):
        if v is None:
            out[i] = last
        else:
            last = v
    nxt = None
    for i in range(len(out) - 1, -1, -1):
        if out[i] is None:
            out[i] = nxt
        else:
            nxt = out[i]
    return [0.0 if v is None else v for v in out]


def smooth_median(vs, w=5):
    """Moving median only - follows real motion, rejects per-frame noise."""
    n = len(vs)
    h = w // 2
    out = []
    for i in range(n):
        s = sorted(vs[max(0, i - h):min(n, i + h + 1)])
        out.append(s[len(s) // 2])
    return out


def time_smooth(ts, vs, half):
    """Median then mean over a window measured in SECONDS, not in samples.

    The median rejects a single failed correlation without letting it drag its
    neighbours; the mean is what buys the root-N noise reduction.
    """
    n = len(vs)
    med = [0.0]*n
    lo = hi = 0
    for i in range(n):
        while lo < n and ts[lo] < ts[i] - half:
            lo += 1
        while hi < n and ts[hi] <= ts[i] + half:
            hi += 1
        w = sorted(vs[lo:hi])
        med[i] = w[len(w)//2] if w else vs[i]
    out = [0.0]*n
    lo = hi = 0
    for i in range(n):
        while lo < n and ts[lo] < ts[i] - half:
            lo += 1
        while hi < n and ts[hi] <= ts[i] + half:
            hi += 1
        w = med[lo:hi]
        out[i] = sum(w)/len(w) if w else med[i]
    return out


def lag1(vs):
    """Lag-1 autocorrelation: how much of a series is real signal vs noise."""
    n = len(vs)
    if n < 8:
        return 0.0
    m = sum(vs) / n
    den = sum((v - m) ** 2 for v in vs)
    if den <= 0:
        return 0.0
    return sum((vs[i] - m) * (vs[i - 1] - m) for i in range(1, n)) / den


def smooth_series(vs):
    """Median, then mean, then a slew limit: spikes, ripple, and lurches."""
    n = len(vs)
    med = []
    h = MEDIAN_W // 2
    for i in range(n):
        w = sorted(vs[max(0, i - h):min(n, i + h + 1)])
        med.append(w[len(w) // 2])
    avg = []
    h = MEAN_W // 2
    for i in range(n):
        w = med[max(0, i - h):min(n, i + h + 1)]
        avg.append(sum(w) / len(w))
    out = list(avg)
    for i in range(1, n):
        d = out[i] - out[i - 1]
        if d > MAX_RESID_STEP:
            out[i] = out[i - 1] + MAX_RESID_STEP
        elif d < -MAX_RESID_STEP:
            out[i] = out[i - 1] - MAX_RESID_STEP
    return out


def accepted(c):
    if c["method"] == "ring":
        return True          # totality: the ring search is well constrained
    return (c["arc"] >= MIN_ARC_DEG and c["n"] >= MIN_INLIERS
            and c["rms"] <= MAX_RMS_PX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="S:/solar-eclipse/out")
    ap.add_argument("--centres", default=None)
    ap.add_argument("--window", default=None,
                    help="output window as WxH in half-resolution px, e.g. 720x720")
    ap.add_argument("--drop-padded", action="store_true",
                    help="drop frames whose window reaches past the sensor")
    ap.add_argument("--pad-tolerance", type=float, default=0.0,
                    help="px of black edge tolerated before a frame is dropped")
    ap.add_argument("--require-disc", action="store_true",
                    help="drop only when the SENSOR cuts the disc, not the window")
    ap.add_argument("--disc-margin", type=float, default=12.0,
                    help="px of clearance demanded around the disc")
    args = ap.parse_args()

    cfgpath = os.path.join(args.out, "configs", "timelapse.json")
    with open(cfgpath) as fh:
        cfg = json.load(fh)
    with open(args.centres or os.path.join(args.out, "diag", "centres.json")) as fh:
        det = json.load(fh)

    W, H = det["width"], det["height"]
    # Join on (file, frame) rather than position: the frame list is filtered
    # after detection runs (blown frames are dropped), so the two lists are not
    # the same length and index alignment would silently mismatch.
    bykey = {(c["file"], c["index"]): c for c in det["centres"]}
    centres = []
    missing = 0
    for i, f in enumerate(cfg["frames"]):
        c = bykey.get((f["file"], f["index"]))
        if c is None:
            missing += 1
            continue
        c = dict(c)
        c["i"] = i
        # Gain identifies a constant-exposure run, which is what the detector's
        # exposure-dependent bias is keyed on.
        c["level"] = round(f.get("gain", 0.0), 4)
        c["dense"] = bool(f.get("dense"))
        centres.append(c)
    if missing:
        print("warning: %d frames have no detection; re-run tl-centres.js" % missing)
    if not centres:
        raise SystemExit("no detections match the frame list")

    # Absolute capture time per frame, needed to anchor the Sun-Moon offset.
    import datetime
    man = json.load(open(os.path.join(args.out, "segments.json")))
    t0 = {f["name"]: datetime.datetime.fromisoformat(f["t0_utc"]) for f in man["files"]}
    fps_of = {f["name"]: f["fps"] for f in man["files"]}
    epoch = min(t0.values())
    for c in centres:
        c["utc"] = ((t0[c["file"]] - epoch).total_seconds()
                    + c["index"] / fps_of[c["file"]])

    # Mid-totality: the midpoint of the unfiltered span. The Sun and Moon centres
    # coincide there, which is what makes it the right anchor for the offset.
    # C3 fell in a recording gap so this is the observed midpoint rather than the
    # true one; the two differ by a few seconds, worth about a pixel.
    tot = [c["utc"] for c in centres if c["state"] == "unfiltered"]
    t_mid = (min(tot) + max(tot)) / 2 if tot else 0.0

    # Group by capture file AND state. A file that spans the filter change has
    # its Sun found two different ways either side of it, and one line through
    # both extrapolates the totality fits back over the crescent frames - which
    # is how 14_13_00 once produced a 1198 px drift across a 60 s capture.
    """
    Totality is ONE group, spanning every capture.

    Everywhere else a capture gets its own intercept, because the mount was
    nudged by hand between captures and a discontinuity there is real. Totality is
    the exception: the scope was touched to take the filter off and again to put
    it back, but not in between, so the three totality captures sit on one
    continuous track. Fitting them separately invented a discontinuity that is not
    in the data - the Moon jumped 33 px in y between 14_14_36 and 14_16_14, where
    the physical drift over that 38 s gap accounts for about 5 px.

    Checked before adopting: a single line through all 149 totality detections
    leaves mean residuals of (-2.6, 3.4) and (1.1, -1.4) px on the two long
    captures. They genuinely are one track.
    """
    groups = {}
    order = []
    for c in centres:
        key = ("TOTALITY", c["state"]) if c["state"] == "unfiltered"             else (c["file"], c["state"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(c)

    """
    One drift rate for the whole session, one intercept per capture.

    Fitting a free slope per capture is what let a bad segment ruin itself. The
    disc fit does not degrade gracefully: as the crescent thins it holds steady
    and then snaps to a different solution, so 14_11_23 tracks smoothly for 800
    frames and then sits 270 px away for the rest. A free slope chases that, and
    a line through both clusters is wrong everywhere.

    The mount does not change its behaviour between captures, though - every
    healthy capture drifts at the same rate. Sharing one slope across the session
    and leaving only the intercept free turns each segment into a single robust
    location estimate, which a median handles even when 40% of the detections
    have jumped somewhere else.
    """
    slopes = []
    for key in order:
        good = [c for c in groups[key] if accepted(c)]
        if len(good) >= 20 and key[1] == "filtered":
            ts = [c["index"] for c in good]
            ax, _, _ = robust_line(ts, [c["cx"] for c in good])
            ay, _, _ = robust_line(ts, [c["cy"] for c in good])
            span = max(ts) - min(ts)
            if span > 200:
                slopes.append((ax, ay))
    if slopes:
        gx = sorted(s[0] for s in slopes)[len(slopes) // 2]
        gy = sorted(s[1] for s in slopes)[len(slopes) // 2]
    else:
        gx = gy = 0.0
    print("shared mount drift: %+0.5f, %+0.5f px/frame  (from %d captures)"
          % (gx, gy, len(slopes)))

    """
    Totality is stabilised per frame, from the corona correlated against itself.

    Everywhere else the disc fit is precise enough to centre each frame on its own
    measurement. Totality has no photosphere to fit, so the detector scores
    candidate centres on the gradient round the Moon's limb, and that scatters
    ~7 px with a lag-1 of only +0.2 to +0.5 - noise, not motion - which is why it
    is not followed and totality rides a straight line instead.

    A line removes drift. It cannot remove the mount's actual wobble, and magnified
    threefold inside a zoom panel that leftover is what makes totality look
    unsteady next to the partials: measured against phase correlation, the line
    sits 0.74 px RMS from where the picture says the frame is.

    Phase correlation does not share the ring search's weakness - the frame carries
    the lunar limb, a hard edge right round the disc, with the prominences on it,
    which is the same signal the drizzle already uses inside a group. Its own noise
    floor is measurable at about 0.18-0.39 px, four times better than the line and
    in the same class as the disc fit that stabilises the partials.

    Each capture is anchored on its MEAN against the line rather than on its first
    frame, so one bad line value cannot displace a whole capture, and the line still
    supplies the relationship BETWEEN captures - right, because the scope was not
    touched during totality.
    """
    ctrack = {}
    try:
        with open(os.path.join(args.out, "diag", "corona_track.json")) as fh:
            for s in json.load(fh)["shifts"]:
                if s.get("ok"):
                    ctrack[(s["file"], s["index"])] = (s["dx"], s["dy"])
        print("corona correlation: %d totality frames measured" % len(ctrack))
    except Exception:
        print("no corona_track.json - totality will ride the modelled line")

    # During totality the detector measures the Moon, which also moves relative
    # to the Sun at the rate corona-drift.js measured.
    dxr = dyr = 0.0
    try:
        with open(os.path.join(args.out, "final", "drift.json")) as fh:
            d = json.load(fh)
        fps = det.get("fps") or 23.3
        dxr = d["driftPxPerSec"]["x"] / 2.0 / fps
        dyr = d["driftPxPerSec"]["y"] / 2.0 / fps
        print("lunar drift %+0.5f, %+0.5f px/frame; mid-totality anchor t=%.1f s"
              % (dxr, dyr, t_mid))
    except Exception:
        pass

    def robust_intercept(vals):
        s = sorted(vals)
        med = s[len(s) // 2]
        dev = sorted(abs(v - med) for v in vals)
        mad = dev[len(dev) // 2]
        cut = max(3.0 * mad, 2.0)
        keep = [v for v in vals if abs(v - med) <= cut]
        if not keep:
            return med, len(vals)
        k = sorted(keep)
        return k[len(k) // 2], len(vals) - len(keep)

    smoothed = [None] * len(cfg["frames"])
    report = []
    drop_idx = set()
    for key in order:
        gs = groups[key]
        name = "%s %s" % (key[0], key[1][:4])
        good = [c for c in gs if accepted(c)] or gs
        # Time variable for this group. Totality is one group spanning three
        # captures, and a capture's frame index restarts at zero, so indexing the
        # model on it would reset the track at every boundary - that is exactly
        # what put a 49 px step there. Absolute seconds are continuous across the
        # whole group; everywhere else the per-capture index is equivalent and
        # keeps the shared mount-drift slope in its native units.
        unified = key[0] == "TOTALITY"
        tvar = (lambda c: c["utc"]) if unified else (lambda c: c["index"])

        """
        Fit on the ORIGINAL sampling cadence, evaluate on every frame.

        One totality level - the short prominence exposure - is resampled to a
        video frame per raw frame so it can be held on screen. That gives it more
        detections than the rest of totality put together, all inside a ten-second
        window, and a least-squares line does not care that they describe the same
        few seconds: it would pivot a three-and-a-half-minute track to fit them.
        Dense frames are therefore left out of the fit and placed by it, which is
        exactly the relationship every other frame already has to the model.
        """
        fit = [c for c in good if not c.get("dense")] or good
        if unified:
            # 149 samples over 3.5 minutes measure the rate better than the
            # session mount drift plus a modelled lunar term can predict it.
            ax, _, _ = robust_line([tvar(c) for c in fit], [c["cx"] for c in fit])
            ay, _, _ = robust_line([tvar(c) for c in fit], [c["cy"] for c in fit])
        else:
            ax, ay = gx, gy
        bx, nx = robust_intercept([c["cx"] - ax * tvar(c) for c in fit])
        by, ny = robust_intercept([c["cy"] - ay * tvar(c) for c in fit])

        """
        Follow the wobble, reject the noise.

        A straight line cannot jitter, but it cannot follow real motion either,
        and the residuals about it are not noise: their lag-1 autocorrelation
        measures +0.75, +0.63 and +0.30 on three sample captures, so the 1-7 px
        sitting on top of the drift is genuine - mount periodic error and seeing.
        Modelling it away leaves the Sun visibly wobbling under a rigid frame.

        So the line is only a baseline now. Residuals from it are clipped to
        reject the cluster a failed fit jumps to, smoothed with a median then a
        mean, and added back. That tracks motion slower than about a fifth of a
        second while per-frame fit noise averages out.
        """
        ts = [c["index"] for c in gs]
        rx, ry = [], []
        for c in gs:
            use = accepted(c)
            dx = c["cx"] - (ax * tvar(c) + bx)
            dy = c["cy"] - (ay * tvar(c) + by)
            if not use or abs(dx) > MAX_RESIDUAL_PX or abs(dy) > MAX_RESIDUAL_PX:
                rx.append(None)
                ry.append(None)
            else:
                rx.append(dx)
                ry.append(dy)
        """
        Frames whose position is not known are dropped, not guessed.

        The scope was physically touched to remove the solar film. That both
        shifts the pointing and leaves it ringing, and the frames during the
        settle are the worst case for the detector: the exposure is still being
        ramped, so they are saturated as well as moving. In 14_13_00 the first
        two unfiltered detections land 148 px and 11 px off where the capture
        settles, and no amount of smoothing rescues them - interpolating just
        spreads a wrong position over the neighbours.

        A dissolve does not fix this either, because the frame underneath is
        genuinely in the wrong place. So any frame whose own detection disagrees
        with the settled model by more than a tolerance is marked untrusted and
        removed from the video. A short gap reads far better than a lurch.
        """
        """
        A capture whose pointing was physically handled is FOLLOWED, not modelled.

        14_05_08 is not a detector failure - both the circle fit and the centroid
        agree the Sun moves hundreds of pixels during it, in steps of 408 px in
        0.86 s. The scope was being handled mid-capture. The frames themselves are
        perfectly good, which is why it looked like a bug.

        A shared-drift line cannot represent that and should not try. Because the
        window is cropped around the Sun, tracking the lurches puts the Sun in the
        same place every frame and the VIDEO comes out smooth - the sensor moved,
        the subject did not. So a centroid-rescued capture takes its positions
        straight from the detections, lightly median-smoothed against per-frame
        noise, with no line and no slew limit to fight the real motion.
        """
        # A capture is followed rather than modelled when its pointing genuinely
        # moved: the detections wander much further than the drift line allows,
        # yet step smoothly from frame to frame. That is a handled or settling
        # scope, not noise - noise gives large steps AND a large excursion.
        # 14_05_08 (handled mid-capture) and 14_17_51 (ringing after the film went
        # back on) both qualify; an ordinary capture drifts a few px and does not.
        follow = all(c["method"] == "centroid" for c in gs)
        if not follow and key[1] != "filtered" and len(gs) >= 12:
            # Excursion measured about the DRIFT LINE, not in absolute position.
            # Ordinary mount drift covers ~40 px over a 60 s capture, so absolute
            # excursion cannot tell a drifting capture from a handled one - it
            # flagged twelve captures including perfectly normal ones. What
            # distinguishes a handled scope is departing from its own drift line.
            # Measured at the ORIGINAL cadence. The step test asks how far the
            # pointing moves between neighbouring samples, and that only means
            # something if the samples are evenly spaced in time. Dense frames sit
            # one raw frame apart instead of twenty, so their steps are twenty
            # times smaller for no physical reason - including them dragged the
            # median under the threshold and sent the whole of totality down the
            # followed path, which then cut 135 frames as "thrashing".
            hs = [c for c in gs if not c.get("dense")] or gs
            dev = [math.hypot(c["cx"] - (ax * tvar(c) + bx),
                              c["cy"] - (ay * tvar(c) + by)) for c in hs]
            steps = sorted(math.hypot(hs[j]["cx"] - hs[j - 1]["cx"],
                                      hs[j]["cy"] - hs[j - 1]["cy"])
                           for j in range(1, len(hs)))
            follow = (max(dev) > FOLLOW_EXCURSION_PX
                      and steps[len(steps) // 2] < FOLLOW_STEP_PX)

        if follow:
            # A wider median where the fits are poor: just after third contact
            # the crescent is thin and arc coverage drops to 60 deg, so single
            # detections are unreliable even though the motion is real.
            xs = smooth_median([c["cx"] for c in gs], FOLLOW_MEDIAN_W)
            ys = smooth_median([c["cy"] for c in gs], FOLLOW_MEDIAN_W)
            for j, c in enumerate(gs):
                smoothed[c["i"]] = (xs[j], ys[j])
            """
            Within a followed capture, drop the frames where the pointing is
            actually thrashing.

            Following works when the scope moves far but continuously - a bump
            that settles, or a slow reposition. It does NOT work when the scope is
            being actively waved: 14_05_08 swings 250 to 370 px between
            consecutive video frames, repeatedly. Each individual frame is a fine
            picture, but no amount of tracking makes that sequence smooth, and
            where the median lags a swing the subject lurches with it.

            So the capture is kept where it is steady and cut where it is not.
            That preserves the recoverable footage instead of losing the whole
            capture, which is what dropping it wholesale did.
            """
            steady = []
            for j in range(len(gs)):
                a = math.hypot(xs[j] - xs[j - 1], ys[j] - ys[j - 1]) if j else 0.0
                b = (math.hypot(xs[j + 1] - xs[j], ys[j + 1] - ys[j])
                     if j + 1 < len(gs) else 0.0)
                steady.append(max(a, b) <= THRASH_PX)

            # Keep only the LONGEST continuous steady stretch. Cutting the
            # thrashing frames alone is not enough: the surviving fragments are
            # separated by the repositioning itself, so the crop still jumps 250
            # to 430 px between them and stitching them back together is no
            # smoother than leaving the thrash in. One continuous piece with a
            # dissolve at each end is the most that can honestly be recovered.
            best_a = best_b = cur_a = 0
            for j in range(len(gs) + 1):
                if j < len(gs) and steady[j]:
                    continue
                if j - cur_a > best_b - best_a:
                    best_a, best_b = cur_a, j
                cur_a = j + 1
            kept = best_b - best_a
            for j, c in enumerate(gs):
                if not (best_a <= j < best_b):
                    drop_idx.add(c["i"])
            print("  %-20s followed directly (%d of %d frames, %d cut as thrashing)"
                  % (name, kept, len(gs), len(gs) - kept))
            report.append((name, len(gs), len(good), len(gs) - kept))
            continue

        """
        Partial phases: centre every frame on its own fit.

        This is the same measurement that places one capture against the next, run
        one level finer - between frames of a single capture instead of between
        captures. Nothing is modelled and nothing is smoothed, so whatever the
        mount did during the capture is removed rather than averaged: drift,
        periodic error, and the hand-sized bumps alike.

        It works here only because the fit is quiet enough to stand on its own.
        The check is a third of a pixel of frame-to-frame scatter against 1 to
        26 px of real motion - a 30:1 margin - and it is worth re-measuring rather
        than assuming if the detector ever changes.

        A frame whose fit is untrustworthy is CUT, not guessed at. That costs
        nothing in continuity here: every surviving frame is centred on the Sun, so
        a cut leaves a small hiccup in time and no jump at all in position, which
        is the opposite of what happens when a modelled track drops a frame.
        """
        if key[1] == "filtered":
            cut = 0
            worst = 0.0
            for c in gs:
                dev = math.hypot(c["cx"] - (ax * tvar(c) + bx),
                                 c["cy"] - (ay * tvar(c) + by))
                if not accepted(c) or dev > FOLLOW_LINE_TOL_PX:
                    drop_idx.add(c["i"])
                    cut += 1
                    continue
                smoothed[c["i"]] = (c["cx"], c["cy"])
                worst = max(worst, dev)
            print("  %-20s stabilised per frame (%d of %d kept, wander %.1f px)"
                  % (name, len(gs) - cut, len(gs), worst))
            report.append((name, len(gs), len(good), cut))
            continue

        untrusted = []
        rx_raw, ry_raw = list(rx), list(ry)
        rx = smooth_series(fill_gaps(rx))
        ry = smooth_series(fill_gaps(ry))

        # Only follow the residual where it is actually motion. Where it is
        # noise, the straight drift line is the better estimate of where the
        # subject was, and following the noise is strictly worse than ignoring it.
        fx = [v for v in rx_raw if v is not None]
        fy = [v for v in ry_raw if v is not None]
        # Never track residuals off the ring search. Its scatter is ~7 px against
        # ~1 px for the circle fit, so even when the lag-1 test passes - which it
        # did for 14_14_36, giving that capture a 1.56 px/frame step against
        # 0.63 for the two totality captures either side of it - following the
        # series injects more error than it removes. Totality is under five
        # seconds of video and real mount wobble over that is sub-pixel.
        ring = all(c["method"] == "ring" for c in gs)
        tracked = (not ring) and (lag1(fx) >= MIN_LAG1 or lag1(fy) >= MIN_LAG1)
        if not tracked:
            rx = [0.0] * len(rx)
            ry = [0.0] * len(ry)

        # Tolerance from this capture's own scatter about its smoothed track.
        dev = sorted(abs(rx_raw[j] - rx[j]) for j in range(len(gs))
                     if rx_raw[j] is not None)
        dev += sorted(abs(ry_raw[j] - ry[j]) for j in range(len(gs))
                      if ry_raw[j] is not None)
        dev.sort()
        med = dev[len(dev) // 2] if dev else 0.0
        tol = max(TRUST_K * med, TRUST_FLOOR_PX)
        """
        Remove per-exposure detection bias inside totality - but only where the
        model is FOLLOWING the detections.

        The ring search locates the Moon's limb from a brightness gradient, and
        where that gradient crosses threshold depends on exposure, so the same
        Moon is reported a few pixels away at a different exposure level: measured
        inside 14_14_36, two levels disagree by 6.6 px in y. Where the residual is
        tracked, that disagreement reaches the render as a nudge at every exposure
        change, and subtracting each level's mean residual removes it. That is
        what this was written for.

        Where the residual is NOT tracked it does the opposite. Totality is placed
        on a bare straight line - `tracked` is false because ring-search scatter is
        noise, so rx and ry are zero - and a line has no per-level offset to
        inherit. Every frame already sits on a perfectly smooth path. Adding a
        constant that changes at each level boundary can then only INTRODUCE a
        step, and it did: 8.2 px going into the prominence level and 11.7 px
        inside 14_14_36, single-frame jumps in a 1120 px window against a normal
        0.72 px per frame. The correction was fighting a problem that the lag-1
        gate had already removed by a different route.

        What is given up is absolute placement: each level's frames sit up to
        8 px from where that level's own detections put them. That is invisible
        here - the window is Sun-fixed, so a viewer sees changes and not offsets,
        and the line is in any case the better estimate of a Moon that genuinely
        moved in a straight line over these three and a half minutes.
        """
        bias = {}
        if unified and tracked:
            runs = {}
            for j, c in enumerate(gs):
                if rx_raw[j] is None or ry_raw[j] is None:
                    continue
                runs.setdefault((c["file"], c["level"]), []).append(
                    (rx_raw[j], ry_raw[j]))
            # The ensemble each level is measured against is the mean of the
            # per-level means, not the mean of every sample. One vote per level is
            # what the correction actually means, and it is the only form that
            # survives the levels being sampled at different rates - the densely
            # resampled prominence level alone carries more frames than the rest
            # of totality, and by sample count it would BE the ensemble.
            mus = [(sum(a for a, _ in vs) / len(vs), sum(b for _, b in vs) / len(vs))
                   for vs in runs.values() if len(vs) >= LEVEL_BIAS_MIN_N]
            gxm = sum(a for a, _ in mus) / len(mus) if mus else 0.0
            gym = sum(b for _, b in mus) / len(mus) if mus else 0.0
            for k, vs in runs.items():
                if len(vs) < LEVEL_BIAS_MIN_N:
                    continue
                dx = sum(a for a, _ in vs) / len(vs) - gxm
                dy = sum(b for _, b in vs) / len(vs) - gym
                m = math.hypot(dx, dy)
                if m > LEVEL_BIAS_MAX_PX:
                    dx *= LEVEL_BIAS_MAX_PX / m
                    dy *= LEVEL_BIAS_MAX_PX / m
                bias[k] = (dx, dy)
                print("  level bias %-14s %-4s  (%+5.2f, %+5.2f) px removed"
                      % (k[0], k[1], dx, dy))

        # Correlation-placed totality frames, anchored per capture on the mean of
        # the line over that same capture.
        corr = {}
        if unified and ctrack:
            byfile = {}
            for j, c in enumerate(gs):
                s = ctrack.get((c["file"], c["index"]))
                if s is not None:
                    byfile.setdefault(c["file"], []).append(
                        (j, s[0], s[1], ax * tvar(c) + bx, ay * tvar(c) + by,
                         c["utc"]))
            for fn in sorted(byfile):
                rows = sorted(byfile[fn], key=lambda r: r[5])
                if len(rows) < CORONA_MIN_N:
                    continue
                n = len(rows)
                ts = [r[5] for r in rows]
                sx = time_smooth(ts, [r[1] for r in rows], CORONA_SMOOTH_S)
                sy = time_smooth(ts, [r[2] for r in rows], CORONA_SMOOTH_S)
                ox = sum(r[3] for r in rows) / n - sum(sx) / n
                oy = sum(r[4] for r in rows) / n - sum(sy) / n
                dev = sorted(math.hypot(sx[i] + ox - rows[i][3],
                                        sy[i] + oy - rows[i][4]) for i in range(n))
                for i, r in enumerate(rows):
                    corr[r[0]] = (sx[i] + ox, sy[i] + oy)
                print("  %-14s placed per frame from the corona (%d frames, "
                      "line differs by %.2f px median, %.2f px p90)"
                      % (fn, n, dev[n // 2], dev[int(0.9 * n)]))

        for j, c in enumerate(gs):
            if j in corr:
                mx, my = corr[j]
                bd = None
            else:
                mx = ax * tvar(c) + bx + rx[j]
                my = ay * tvar(c) + by + ry[j]
                bd = bias.get((c["file"], c.get("level")))
            if bd:
                mx += bd[0]
                my += bd[1]
            if key[1] == "unfiltered":
                # Convert the Moon track into a Sun track. The detector has no
                # photosphere to fit during totality and measures the Moon, but
                # the corona - the thing being watched - is attached to the Sun.
                # The two centres coincide at greatest eclipse and separate to at
                # most (Rmoon - Rsun) at each contact, about 9 px here, moving
                # very nearly linearly in between at the measured differential
                # rate. Subtracting that offset frames the Sun instead.
                dt = c["utc"] - t_mid
                mx -= dxr * dt * (det.get("fps") or 23.3)
                my -= dyr * dt * (det.get("fps") or 23.3)
            smoothed[c["i"]] = (mx, my)
            # The trust check asks whether the frame's own DETECTION agrees with
            # where the model put it. A correlation-placed frame does not depend on
            # that detection at all, so a bad ring search is no longer a reason to
            # drop it - the picture itself said where the frame was.
            if j not in corr and (rx_raw[j] is None or ry_raw[j] is None
                                  or abs(rx_raw[j] - rx[j]) > tol
                                  or abs(ry_raw[j] - ry[j]) > tol):
                untrusted.append(c["i"])
        """
        Drop an untrusted frame only when the capture around it is healthy.

        The point of the check is to catch the handful of frames where the scope
        was knocked and the position is genuinely unknown. But when MOST of a
        capture fails the check, the frames are not the problem - the detector is,
        and dropping the lot throws away good footage for a fault of mine.
        14_05_08 lost all 71 of its frames that way, and 14_11_23 lost 27.

        So a mostly-failed capture is kept whole and placed by its robust median
        instead. That gives up the sub-pixel residual tracking for that capture -
        it rides the shared drift line only - which is a much smaller loss than a
        two-second hole in the video.
        """
        if len(untrusted) > MOSTLY_FAILED * len(gs):
            # A mostly-failed capture is only worth keeping if its detections at
            # least agree with each other - then the fault is the trust threshold
            # and the median is a fair estimate. When they disagree wildly the
            # position is genuinely unknown: 14_05_08 scatters over 595 px in x
            # and 444 px in y, so its median is the median of noise, and placing
            # the capture there put the Sun 35 px off and made the video lurch
            # going into it. Better a short gap than a confidently wrong position.
            # Scatter about the fitted track, NOT raw position spread. A long
            # group legitimately covers a lot of ground - unified totality spans
            # 3.5 minutes and 210 px of real Moon motion - and judging it on raw
            # spread declared the whole of totality unplaceable and dropped it.
            # A None residual means the detection was too far out to measure at
            # all - that is a failure, not agreement, and counting it as zero
            # scatter is how 14_05_08 came back as "spread 0 px" and put the
            # 10 s lurch back in. Only measurable residuals count, and a group
            # with too few of them is unplaceable by definition.
            rr = sorted(math.hypot(rx_raw[j], ry_raw[j]) for j in range(len(gs))
                        if rx_raw[j] is not None and ry_raw[j] is not None)
            spread = rr[int(len(rr) * 0.9)] if len(rr) >= 0.25 * len(gs)                 else float("inf")
            if spread <= UNPLACEABLE_SPREAD_PX:
                print("  %-20s detector failed on %d/%d (spread %.0f px) - kept"
                      % (name, len(untrusted), len(gs), spread))
            else:
                print("  %-20s detector failed on %d/%d, spread %.0f px - dropped"
                      % (name, len(untrusted), len(gs), spread))
                drop_idx.update(c["i"] for c in gs)
        else:
            drop_idx.update(untrusted)
        report.append((name, len(gs), len(good), max(nx, ny)))

    # Any frame with no detection inherits its neighbour's position.
    last = None
    for i in range(len(smoothed)):
        if smoothed[i] is None:
            smoothed[i] = last if last else (W / 2.0, H / 2.0)
        last = smoothed[i]

    # Nothing may fall outside the sensor; a model that says otherwise is wrong.
    smoothed = [(min(max(cx, 0.0), float(W)), min(max(cy, 0.0), float(H)))
                for (cx, cy) in smoothed]

    print("per-capture Sun track (half-resolution px):")
    print("  %-20s %5s %5s %9s" % ("file / state", "frm", "used", "outliers"))
    for r in report:
        print("  %-20s %5d %5d %9d" % r)

    xs = [p[0] for p in smoothed]
    ys = [p[1] for p in smoothed]
    print("\nSun excursion: x %.0f..%.0f (%.0f px), y %.0f..%.0f (%.0f px) of %dx%d"
          % (min(xs), max(xs), max(xs) - min(xs),
             min(ys), max(ys), max(ys) - min(ys), W, H))

    # Largest Sun-centred window that keeps padding rare. Holding the Sun fixed
    # while it wanders the sensor means the window sometimes reaches past the
    # edge; padding there is honest, clipping the Sun is not.
    def padded_fraction(ow, oh):
        bad = 0
        for (cx, cy) in smoothed:
            if (cx - ow / 2 < 0 or cx + ow / 2 > W
                    or cy - oh / 2 < 0 or cy + oh / 2 > H):
                bad += 1
        return bad / len(smoothed)

    rSun = det.get("rSun") or 279.0
    if args.window:
        ow, oh = (int(v) for v in args.window.lower().split("x"))
    else:
        # Size the window to the subject, not to what happens to be clip-free.
        ow = int(round(2 * CORONA_RADII * rSun / SIZE_STEP)) * SIZE_STEP
        oh = int(round(ow * 9 / 16 / SIZE_STEP)) * SIZE_STEP
        while ow > 2 * MIN_RADII * rSun and padded_fraction(ow, oh) > MAX_PADDED_FRACTION:
            ow -= SIZE_STEP
            oh = int(round(ow * 9 / 16 / SIZE_STEP)) * SIZE_STEP
    print("output window %dx%d (%dx%d full-res) = %.2f R wide, %.2f R tall; "
          "%.1f%% of frames reach past an edge"
          % (ow, oh, ow * 2, oh * 2, ow / 2 / rSun, oh / 2 / rSun,
             100 * padded_fraction(ow, oh)))

    for f, (cx, cy) in zip(cfg["frames"], smoothed):
        f["cx"] = round(cx, 3)
        f["cy"] = round(cy, 3)

    if drop_idx:
        by_file = {}
        for i in sorted(drop_idx):
            f = cfg["frames"][i]
            by_file[f["file"]] = by_file.get(f["file"], 0) + 1
        print("untrusted position, dropped %d frames: %s" % (
            len(drop_idx), ", ".join("%s x%d" % (k, v) for k, v in sorted(by_file.items()))))
        cfg["frames"] = [f for i, f in enumerate(cfg["frames"]) if i not in drop_idx]
        smoothed = [p for i, p in enumerate(smoothed) if i not in drop_idx]

    if args.drop_padded:
        # A Sun-fixed window sometimes reaches past the sensor on the partial
        # phases, where the Sun sat near a corner. Rather than show a black wedge
        # against a sky that is not quite black, drop those frames: the partial
        # phase is slow and repetitive, so losing some of it costs nothing, and
        # every totality frame survives.
        """
        Tolerate a little black rather than lose the frame.

        A window held on the Sun sometimes reaches past the sensor, and the strict
        rule dropped every such frame. But the amount is usually tiny - a quarter
        of them reach over by under 10 px on a 1100 px window, under 1% of the
        width - and it happens in the dark sky at a corner, where a few black
        pixels are far less noticeable than a gap in the sequence. Only frames
        that would show an obvious black wedge are worth dropping.
        """
        keep, lost = [], {}
        # Two possible criteria.
        #
        # --require-disc asks only that the SENSOR contained the whole disc. This
        # is a timelapse, not a stack: a black wedge in the corner of the field is
        # cosmetic, whereas a frame where the sensor cut into the disc is missing
        # subject and cannot be used. Under this rule the window is allowed to
        # hang off the sensor as far as it likes and the renderer pads it.
        #
        # Otherwise the window itself must sit inside the sensor, give or take
        # --pad-tolerance.
        rdisc = (det.get("rMoon") or det.get("rSun") or 292.0) + args.disc_margin
        for f in cfg["frames"]:
            if args.require_disc:
                over = max(0.0, rdisc - f["cx"], f["cx"] + rdisc - W,
                           rdisc - f["cy"], f["cy"] + rdisc - H)
            else:
                over = max(0.0, ow / 2 - f["cx"], f["cx"] + ow / 2 - W,
                           oh / 2 - f["cy"], f["cy"] + oh / 2 - H)
                over = max(0.0, over - args.pad_tolerance)
            if over > 0:
                lost[f["file"]] = lost.get(f["file"], 0) + 1
                continue
            keep.append(f)
        n_tot = sum(1 for f in cfg["frames"] if f["state"] == "unfiltered")
        k_tot = sum(1 for f in keep if f["state"] == "unfiltered")
        print("dropped %d of %d frames (%.1f%%); totality kept %d of %d"
              % (len(cfg["frames"]) - len(keep), len(cfg["frames"]),
                 100 * (len(cfg["frames"]) - len(keep)) / len(cfg["frames"]),
                 k_tot, n_tot))
        for k in sorted(lost):
            print("    %-14s %3d" % (k, lost[k]))
        cfg["frames"] = keep

    cfg["outW"] = ow
    cfg["outH"] = oh
    cfg["sunFixed"] = True
    with open(cfgpath, "w") as fh:
        json.dump(cfg, fh)
    print("%d frames, %.1f s at %d fps" % (len(cfg["frames"]),
                                           len(cfg["frames"]) / cfg["fps"], cfg["fps"]))
    print("updated %s" % cfgpath)


if __name__ == "__main__":
    main()
