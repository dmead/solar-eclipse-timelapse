"""Stage F planner — choose timelapse frames and their exposure normalization.

Stage A already sampled every Nth frame of every capture and recorded its
brightness, so the timelapse costs no extra measurement pass: the same sample
positions become the video frames, and the recorded statistics become the gains.

The normalization is per SEGMENT, not per frame, and that is the important part.
Normalizing every frame to a constant brightness would erase the eclipse — the sky
genuinely darkens as totality approaches, and that is the thing worth watching.
What must be removed is only the operator's own exposure changes. Stage A already
knows where those are: a stable segment is by definition an interval of constant
camera settings, so it gets one gain, while a transition segment is the operator
actively riding the exposure and gets a per-frame gain that cancels the ride.

Gains are chained across segment boundaries using background level as the
exposure proxy, and only across boundaries. Background level is not a pure
measure of exposure — it also falls as the eclipse deepens and less sunlight is
scattered — so it is trustworthy only over the fraction of a second spanning a
boundary, where the sky is unchanged and the camera is not. Comparing a whole
segment's average would confuse the sky darkening for an exposure change and
drive the gain up until the crescent blew out.

Filtered and unfiltered runs are chained separately and anchored independently:
when the filter comes off the subject changes from photosphere to corona, and no
gain relates the two. That boundary is a hard cut in the video, which is what it
looked like in person.

    python gen_timelapse.py --out S:/solar-eclipse/out

Writes `configs/timelapse.json` for pjsr/tl-frames.js.
"""

import argparse
import json
import os

import numpy as np

FULL_SCALE = 65535.0

# Rendered brightness the anchor segment's p99 is mapped to, per state. The
# photosphere is allowed to sit brighter than the corona: it is a white disc and
# should read as one, while the corona needs headroom for the inner regions.
TARGET = {"filtered": 0.88, "unfiltered": 0.78}

# Transitions sampled this thinly cannot be normalized; they are dropped.
MIN_TRANSITION_SAMPLES = 3

# Brightest a transition frame may render, relative to the stable target.
TRANSITION_CEILING = 1.15

"""
On-screen time given to the shortest-exposure stable level inside totality.

That level is the prominence exposure - short enough that the chromosphere is not
blown out, which is what makes the prominences readable - and the operator was on
it only briefly before opening up for the corona. Sampled at the light curve's
own cadence it produced twelve video frames, four tenths of a second, and it went
past before there was anything to see.

Nothing else in the video needs this. The partial phases are slow and repetitive
and the other totality levels each run for seconds already; it is specifically the
one level whose subject is unique and whose footage is short.
"""
SLOWMO_LEVEL_S = 12.0

# Saturation a frame may carry before it is dropped, by state.
#
# With the filter on nothing should clip at all, so anything over a percent is
# the operator having overshot, and no gain can un-clip it.
#
# During totality the chromosphere and prominences legitimately clip - a 20 px
# ring at this solar radius is ~0.9% of the frame - but the exposure ramp right
# after the filter comes off runs to 10.8%, and those frames render as a huge
# soft blob with no detail in them at all. Same 2% bound the corona ladder uses.
MAX_SATFRAC = {"filtered": 0.01, "unfiltered": 0.02}


def tune(out_dir, log=None):
    """Resolve selection and dwell settings from the config."""
    global TARGET, MAX_SATFRAC, TRANSITION_CEILING, STACK_MAX
    global FLATTEN_MIN_AREA, FLATTEN_SAT, FLATTEN_MAX
    global SLOWMO_LEVEL_S, RESOLVE_S, BEADS_S, CORONA_S, BEAD_STACK
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ecl.params import load

    P = load(out_dir, create=False)
    TARGET = {"filtered": P.get("select.target_filtered", TARGET["filtered"]),
              "unfiltered": P.get("select.target_unfiltered", TARGET["unfiltered"])}
    MAX_SATFRAC = {
        "filtered": P.get("select.max_satfrac_filtered", MAX_SATFRAC["filtered"]),
        "unfiltered": P.get("select.max_satfrac_unfiltered",
                            MAX_SATFRAC["unfiltered"])}
    TRANSITION_CEILING = P.get("select.transition_ceiling", TRANSITION_CEILING)
    STACK_MAX = P.get("select.stack_max", STACK_MAX)
    FLATTEN_MIN_AREA = P.area("select.min_crescent_r2", FLATTEN_MIN_AREA)
    FLATTEN_SAT = P.get("select.crescent_sat", FLATTEN_SAT)
    FLATTEN_MAX = P.get("select.flatten_max", FLATTEN_MAX)
    SLOWMO_LEVEL_S = P.get("dwell.prominence_s", SLOWMO_LEVEL_S)
    RESOLVE_S = P.get("dwell.resolve_s", RESOLVE_S)
    BEADS_S = P.get("dwell.beads_s", BEADS_S)
    BEAD_STACK = P.get("select.bead_stack", BEAD_STACK)
    CORONA_S = P.get("dwell.corona_s", CORONA_S)
    if log:
        log(f"  dwells: resolve {RESOLVE_S}s, beads {BEADS_S}s, "
            f"prominence {SLOWMO_LEVEL_S}s, corona {CORONA_S}s")
    return P


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def level(point):
    """Background level above the black floor, for one sample."""
    return max(point["med"] - point["black"], 1.0)


# Crescent smaller than this, in px, and the measurement cannot be separated from
# sky well enough to normalize on. Near second and third contact.
FLATTEN_MIN_AREA = 2000
# Above this the crescent is clipped in the raw data, so its level says nothing
# about the exposure. Three captures are entirely in this state.
FLATTEN_SAT = 0.97
# Never rescale a frame by more than this. A correction this large means the
# measurement is wrong, not the gain.
FLATTEN_MAX = 2.0


def crescent_level(plane):
    """Median level of the lit crescent, and its area in px.

    Thresholded halfway between the sky and the peak rather than at a fixed
    percentile, so it follows the crescent down as the Moon covers it. A
    percentile threshold stops being the photosphere once the crescent is thinner
    than the percentile itself and silently becomes a sky measurement.
    """
    sky = float(np.median(plane))
    peak = float(np.percentile(plane, 99.99))
    if peak <= sky * 1.5:
        return 0.0, 0
    m = plane > sky + 0.5 * (peak - sky)
    n = int(m.sum())
    return (float(np.median(plane[m])) if n else 0.0), n


def flatten_captures(frames, log=print):
    """Hold the rendered photosphere level constant within each capture.

    The target is the rendered level - level times gain - and not the raw level.
    Aiming at the raw level would undo the segment gain wherever a capture
    genuinely contains an exposure change: 14_06_45 has one, and correcting the
    raw level there turned a +13.6% step into -39.2% by cancelling the gain that
    was handling it correctly.

    Where the crescent cannot be measured the correction is HELD from the nearest
    frame that could be, never reset to 1. Resetting would put a seam exactly
    where measurable meets unmeasurable, which is the thing this exists to remove.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ecl.source import open_source

    want = [f for f in frames if f["state"] == "filtered"
            and not f.get("resolve") and not f.get("bead")]
    if not want:
        return

    # Dense and hold frames repeat raw frames, so measure each one once.
    uniq = sorted({(f["src"], f["index"]) for f in want})
    log("  measuring the crescent on %d raw frame(s) for %d video frame(s)"
        % (len(uniq), len(want)))
    meas, src, cur = {}, None, None
    try:
        for path, idx in uniq:
            if path != cur:
                if src is not None:
                    src.close()
                src = open_source(path)
                cur = path
            meas[(path, idx)] = crescent_level(src.green(idx).astype(np.float32))
    finally:
        if src is not None:
            src.close()

    by = {}
    for f in want:
        by.setdefault(f["file"], []).append(f)

    n_fixed = 0
    for name in sorted(by, key=lambda k: frames.index(by[k][0])):
        v = by[name]
        lvl, ok = [], []
        for f in v:
            phot, area = meas[(f["src"], f["index"])]
            lvl.append(phot * f["gain"])
            ok.append(area >= FLATTEN_MIN_AREA and phot < FLATTEN_SAT and phot > 0)
        if sum(ok) < 3:
            log("  %-14s no measurable crescent - left alone" % name)
            continue
        good = [lvl[i] for i in range(len(v)) if ok[i]]
        target = median(good)
        corr = [1.0] * len(v)
        for i in range(len(v)):
            if ok[i] and lvl[i] > 0:
                corr[i] = min(max(target / lvl[i], 1.0 / FLATTEN_MAX), FLATTEN_MAX)
        # Carry the nearest measured correction across the unmeasurable frames.
        idx_ok = [i for i in range(len(v)) if ok[i]]
        for i in range(len(v)):
            if not ok[i]:
                corr[i] = corr[min(idx_ok, key=lambda j: abs(j - i))]
        before = max(lvl) / max(min(lvl), 1e-9) - 1.0
        after = (max(l * c for l, c in zip(lvl, corr))
                 / max(min(l * c for l, c in zip(lvl, corr)), 1e-9) - 1.0)
        for f, c in zip(v, corr):
            f["gain"] = round(f["gain"] * c, 8)
        n_fixed += 1
        log("  %-14s spread %5.1f%% -> %4.1f%%   (%d of %d measurable)"
            % (name, 100 * before, 100 * after, sum(ok), len(v)))
    log("  flattened %d capture(s)" % n_fixed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="S:/solar-eclipse/out")
    ap.add_argument("--stride", type=int, default=1,
                    help="take every Nth light-curve sample (1 = all)")
    ap.add_argument("--fps", type=int, default=30)
    # Frames land on an SSD, not on S:. S: is a spinning SATA HDD at 117 MB/s and
    # a render already reads 551 GB off it; writing 1660 PNGs back to the same
    # head makes it seek between the read and write zones on every frame.
    ap.add_argument("--frames-dir", default="D:/eclipse-work/tl")
    args = ap.parse_args()
    tune(args.out, log=print)

    with open(os.path.join(args.out, "lightcurve.json")) as fh:
        lc = json.load(fh)
    with open(os.path.join(args.out, "segments.json")) as fh:
        man = json.load(fh)

    """
    The bead window is MEASURED, by `ecl.beadwindow`, not inferred here.

    Saturation cannot find it. The diamond ring and the beads are both clipped
    photosphere and the clipped area falls smoothly through both - satfrac reads
    0.00025 at f1160 and 0.00018 at f1180, on either side of the moment the ring
    breaks up. Drawing the split there put the ten-second dwell on the ring: a
    single overexposed blob, held while the part worth watching went past at
    speed.

    What separates them is the shape of the clipped region, and that needs a
    frame read, which this stage does not do. So it is measured in its own pass
    and read back here. Without the file the old saturation split still applies,
    which is wrong in the same way it always was but no worse.
    """
    beads = {}
    bpath = os.path.join(args.out, "diag", "beads.json")
    if os.path.exists(bpath):
        with open(bpath) as fh:
            beads = json.load(fh)
        for k, v in beads.items():
            print("  beads measured %s f%d-%d (%.2f s of footage)"
                  % (k, v["lo"], v["hi"], v["seconds"]))
    else:
        print("  no diag/beads.json - run ecl.beadwindow; "
              "falling back to the saturation split")

    points_by_file = {f["name"]: f["points"] for f in lc["files"]}
    path_by_file = {f["name"]: f["path"] for f in lc["files"]}

    # Chronological list of segments with the samples that fall inside each.
    segs = []
    for f in man["files"]:
        pts = points_by_file[f["name"]]
        for s in f["segments"]:
            inside = [p for p in pts
                      if s["start"] <= p["i"] < s["start"] + s["count"]]
            if not inside:
                continue
            segs.append({
                "file": f["name"], "path": path_by_file[f["name"]],
                "state": s["state"], "kind": s["kind"],
                "start": s["start"], "count": s["count"],
                "points": inside, "fps": f["fps"],
                # Levels at the two edges of the segment: the only places a
                # comparison with a neighbour is about the camera and not the sky.
                "e_in": level(inside[0]),
                "e_out": level(inside[0] if s["kind"] == "transition" else inside[-1]),
                "p99": median([p["p99"] for p in inside]),
            })

    # Split into runs of like state, dropping the blown-out frames at the filter
    # change — those are wholly clipped and belong to Stage E, not the video.
    runs, cur = [], []
    for s in segs:
        # A one- or two-sample transition carries no reliable level to normalize
        # against, and rendering it produces a single black or blown flash.
        if s["kind"] == "transition" and len(s["points"]) < MIN_TRANSITION_SAMPLES:
            continue
        if s["kind"] == "blown":
            if cur:
                runs.append(cur)
                cur = []
            continue
        if cur and cur[-1]["state"] != s["state"]:
            runs.append(cur)
            cur = []
        cur.append(s)
    if cur:
        runs.append(cur)

    frames = []
    for run in runs:
        # Anchor on the longest stable segment, then propagate outward using only
        # boundary-to-boundary ratios.
        ai = max(range(len(run)), key=lambda i: (run[i]["kind"] == "stable", run[i]["count"]))
        g = [0.0] * len(run)
        g[ai] = 1.0
        for k in range(ai + 1, len(run)):
            g[k] = g[k - 1] * run[k - 1]["e_out"] / run[k]["e_in"]
        for k in range(ai - 1, -1, -1):
            g[k] = g[k + 1] * run[k + 1]["e_in"] / run[k]["e_out"]

        anchor = run[ai]
        # Gains are consumed by tl-frames.js, which normalizes samples to [0,1]
        # before applying them, so the full-scale factor belongs here: a pixel at
        # the anchor's p99 must come out at TARGET, not TARGET/65535.
        scale = TARGET[anchor["state"]] * FULL_SCALE / anchor["p99"]

        """
        During totality, normalize each stable segment directly.

        The chained gains are right for the partials, where the subject genuinely
        changes brightness and only the operator's exposure steps should be
        removed. Totality is the opposite case: the corona's real brightness is
        essentially constant over the four minutes it is visible, so ANY change
        between segments is the operator riding the exposure, and all of it should
        go. Chaining accumulated error instead - 14_14_36 rendered at 0.31 to 0.44
        against 0.67 for the segment before it and 0.80 for the one after, which
        reads as the corona pulsing.

        Anchoring each stable unfiltered segment to the target directly removes
        that. It would be wrong for the filtered phases, which is why it is not
        applied there.
        """
        totality = run[ai]["state"] == "unfiltered"

        gains = []
        for k, s in enumerate(run):
            gk = g[k] * scale
            if totality and s["kind"] == "stable":
                gk = TARGET[s["state"]] * FULL_SCALE / s["p99"]
            gains.append(gk)

        """
        Find the prominence level by MEASUREMENT, not by naming a capture.

        Gain is the inverse of exposure once the normalization is applied - every
        stable totality segment is mapped to the same rendered brightness, so the
        one needing the most gain is the one that was shot shortest. Here that is
        20x clear of the rest, and it is the run right after second contact.
        """
        slow = None
        if totality:
            stable = [k for k, s in enumerate(run) if s["kind"] == "stable"]
            if stable:
                slow = max(stable, key=lambda k: gains[k])

        """
        The second-contact resolve: keep it, and slow it down.

        The leading segments of a totality run are the filter coming off - the
        last of the photosphere breaking into beads and going out. Measured on
        14_13_00 the saturated limb goes from a closed 360 deg ring to a 34 deg
        arc over 279 raw frames, and the annulus median falls 50x while it does.
        That is the diamond ring and Baily's beads, and it was being thrown away:
        every one of those frames clips more than MAX_SATFRAC allows, so the video
        cut from the last filtered frame straight to a corona already three
        seconds old.

        They are marked rather than merely kept, because three later decisions
        need to know: the saturation gate must not drop them, they must not be
        stacked (the beads change inside one drizzle group), and at the light
        curve's own 20-frame sampling the whole sequence would be fourteen video
        frames - half a second - so they get their own screen time.

        The bound is the saturation itself, walked forward from the start of the
        run until a segment comes in under the normal cap. No contact time, no
        capture named in a constant.
        """
        resolve = set()
        if totality:
            for k, s in enumerate(run):
                if median([p["satfrac"] for p in s["points"]]) <= MAX_SATFRAC[s["state"]]:
                    break
                resolve.add(k)

        """
        Three dwells inside totality, because 24 s of it was spent in the wrong
        places.

        Measured on the previous cut: 12.0 s on the prominence level, 8.0 s on the
        resolve, and 4.5 s on the corona itself - while 2,583 raw frames of corona
        sat unused, because the light curve samples one frame in twenty and the
        corona segments were emitted at that cadence. The classic totality view
        was the shortest thing in totality.

        So the corona and the bead phase each get an explicit screen time, the same
        way the prominence level already did. Where a dwell has more raw frames
        than it needs it thins them and every video frame is a different exposure;
        where it has fewer it repeats them, and the resolve/beads split is drawn
        at the saturation where the plateau has become a chain.
        """
        # The corona proper: stable totality segments that are neither the
        # resolve nor the prominence level.
        corona = {k for k, s in enumerate(run)
                  if totality and s["kind"] == "stable" and k not in resolve
                  and k != slow} if totality else set()

        """
        The bead window CUTS ACROSS segments, so screen time is allocated to raw
        frame ranges rather than to whole segments.

        Measured, the beads run f1189-1255, which is inside the prominence level
        and nowhere near the resolve that ends at f1170. Nothing about the
        exposure changes there - the operator had already settled - so no segment
        boundary marks it and none ever will. Each segment is therefore split
        into up to three parts: whatever precedes the window, the window itself,
        and whatever follows. The middle part is sampled to fill its share of
        BEADS_S; the outer parts keep the treatment the segment would have had,
        sharing that phase's own budget between them.
        """
        def bead_span(seg):
            w = beads.get(seg["file"])
            if not w:
                return None
            lo = max(seg["start"], w["lo"])
            hi = min(seg["start"] + seg["count"], w["hi"] + 1)
            return (lo, hi) if hi - lo >= 2 else None

        spans = {k: bead_span(s) for k, s in enumerate(run)}
        n_bead = sum(v[1] - v[0] for v in spans.values() if v)

        def outside(k):
            """Raw frames of segment k that are NOT in the bead window."""
            s = run[k]
            n = s["count"]
            return n - (spans[k][1] - spans[k][0] if spans[k] else 0)

        n_res = sum(outside(k) for k in resolve)
        n_cor = sum(outside(k) for k in corona)
        n_slow = outside(slow) if slow is not None else 0

        for k, s in enumerate(run):
            gk = gains[k]
            s_lo, s_hi = s["start"], s["start"] + s["count"]
            sp = spans[k]
            parts = ([(s_lo, sp[0], False), (sp[0], sp[1], True), (sp[1], s_hi, False)]
                     if sp else [(s_lo, s_hi, False)])

            pts, hold = [], False
            for lo, hi, is_bead in parts:
                if hi - lo < 1:
                    continue
                if is_bead:
                    got = stretch_points(s, args.fps,
                                         BEADS_S * (hi - lo) / max(n_bead, 1),
                                         lo, hi)
                    for p in got:
                        p["_bead"] = True
                    pts += got
                elif k in resolve:
                    # The GAIN still comes from the segment's own kind. Taking a
                    # single segment gain across the resolve looked right until
                    # its last frame sat at 0.65 against the 27.31 of the
                    # prominence level one raw frame later - a 42x step, because
                    # the operator was ramping right through here and a
                    # transition has to be followed point by point. The branch
                    # below does that, so this does not repeat it.
                    pts += stretch_points(s, args.fps,
                                          RESOLVE_S * (hi - lo) / max(n_res, 1),
                                          lo, hi)
                elif k == slow:
                    pts += dense_points(s, args.fps,
                                        SLOWMO_LEVEL_S * (hi - lo) / max(n_slow, 1),
                                        lo, hi)
                elif k in corona:
                    pts += stretch_points(s, args.fps,
                                          CORONA_S * (hi - lo) / max(n_cor, 1),
                                          lo, hi)
                    hold = True
                else:
                    pts += [p for p in s["points"] if lo <= p["i"] < hi]

            dense = k == slow
            if s["kind"] == "stable":
                # One gain for the whole segment: the camera did not change, so
                # whatever brightness change happens here is the eclipse itself.
                for p in pts:
                    frames.append(mkframe(s, p, gk, resolve=k in resolve,
                                          hold=hold, dense=dense,
                                          bead=p.get("_bead", False)))
            else:
                # Exposure is moving inside this segment; follow it frame by frame
                # so the operator's ramp does not show up as a flare in the video.
                # A transition inside totality is anchored to the stable segment
                # that follows it, so the ramp lands on the same brightness the
                # rest of totality holds instead of on the chain's estimate.
                anchor = gk
                if totality:
                    nxt = next((run[m] for m in range(k + 1, len(run))
                                if run[m]["kind"] == "stable"), None)
                    if nxt:
                        anchor = (TARGET[nxt["state"]] * FULL_SCALE / nxt["p99"]
                                  * nxt["e_in"] / s["e_in"])
                for p in pts:
                    gp = anchor * s["e_in"] / level(p)
                    # A transition frame must never render brighter than the
                    # stable frames it sits between. The exposure proxy moves fast
                    # through a ramp and the estimate overshoots badly at the end
                    # of one - single frames were landing at 3.5 to 4.9 against a
                    # 0.78 target, which is a white flash for one frame. Capping
                    # on the frame's own p99 bounds that directly.
                    cap = TRANSITION_CEILING * TARGET[s["state"]] * FULL_SCALE / max(p["p99"], 1)
                    frames.append(mkframe(s, p, min(gp, cap),
                                          resolve=k in resolve, hold=hold,
                                          dense=dense,
                                          bead=p.get("_bead", False)))

    frames.sort(key=lambda p: (p["file"], p["index"]))
    # Thinning is for previewing the whole video quickly; the densely resampled
    # level is exempt, since decimating it is the one thing it exists to undo.
    if args.stride > 1:
        frames = [f for i, f in enumerate(frames)
                  if f.get("dense") or i % args.stride == 0]

    """
    Give the totality frames a drizzle group.

    Prominences are Halpha, so they sit almost entirely in R, and R is one sample
    per 2x2 CFA site - 3.43 arcsec/px, half the linear resolution of luminance and
    the worst-sampled signal in the capture. Only drizzle recovers that, and
    drizzle needs several frames with sub-pixel dither. The unaligned mount
    supplied the dither for free: measured across a group the pointing moves 0.89
    px, which lands the frames in 8 or 9 of the 16 sub-pixel cells.

    The group is the frames between this video frame and the next, so groups
    tile the capture without overlapping. Two things bound it. It may not run past
    the end of its SEGMENT, because segments are constant-exposure by construction
    and a group spanning an exposure change would average two brightnesses. And it
    is capped, because the group is also an effective shutter: 20 frames is 0.86 s,
    over which the Moon advances 0.24 px, well under the seeing.

    Partials are stacked too, on the same terms. They were left out at first on
    the grounds that a hard white-light limb is not where the detail is, and that
    it means reading all 446 GB rather than the totality slices. Neither holds up.
    The panels magnify a sunspot to 3x and a cusp to as much as 7x, and at those
    magnifications an unstacked frame is being enlarged past its own noise - there
    is nothing under it but a bicubic interpolation of the superpixel grid. And the
    read is not the problem: 1164 filtered frames at 20 raw frames each is 386 GB,
    which S: delivers in about six minutes. The cost is CPU in the renderer, not
    I/O.
    """
    for i, f in enumerate(frames):
        f["stack"] = 1
        if f.get("bead"):
            # A SHORT stack, and this has to be said HERE rather than left to the
            # contact-window rule below. The measured bead window sits inside the
            # prominence level, so these frames are also `dense` and would take
            # its overlapping twenty-frame groups - averaging the beads into the
            # smooth arc the whole dwell exists to show breaking up. They are also
            # 19 raw frames past the state change, so the BEAD_FRAMES window does
            # not reach them.
            #
            # It used to be stack 1 outright, on the grounds that the beads change
            # inside a group's span. True of a 20-frame group, which is 0.86 s;
            # not true of BEAD_STACK frames, which is 0.13 s against a 2.87 s
            # window. Raw frames here are single exposures at a short shutter and
            # they are visibly grainy - and the dwell repeats each one several
            # times, which freezes that grain in place where the eye can study it.
            f["stack"] = max(1, min(BEAD_STACK, f["seg_end"] - f["index"]))
            continue
        if f.get("resolve"):
            # Never stacked, and not merely by the gap rule happening to give 1:
            # the whole sequence is the beads changing inside a group's span.
            continue
        if f.get("dense") or f.get("hold"):
            # Dense and hold frames sit one raw frame apart, so the gap-to-next rule
            # would hand them a group of one and undo the drizzle exactly where it
            # matters most. Their groups OVERLAP instead: consecutive frames share
            # nineteen of their twenty raw frames. Nothing about drizzle requires
            # groups to be disjoint - tiling was only ever a way to use each raw
            # frame once - and overlapping keeps every frame at full stack depth.
            f["stack"] = max(1, min(STACK_MAX, f["seg_end"] - f["index"]))
            continue
        nxt = frames[i + 1] if i + 1 < len(frames) else None
        gap = (nxt["index"] - f["index"]
               if nxt is not None and nxt["file"] == f["file"] else STACK_MAX)
        f["stack"] = max(1, min(STACK_MAX, gap, f["seg_end"] - f["index"]))

    """
    Baily's beads must NOT be stacked.

    Everything above exists to average twenty raw frames into one, which is the
    right trade everywhere except here. The beads and the diamond ring are the
    one subject in this capture that evolves faster than a group spans: a group
    is 0.86 s and the beads change visibly inside that, so stacking averages
    them into the smooth arc that `ser-frames.js` was written to avoid. The
    render loses the drizzle's noise advantage across the contact window, and
    that is the correct trade - a sharp bead is the picture, a clean smooth arc
    is not.

    Found from the state change rather than from a contact time. The filter came
    off at 19:13:35 and second contact was ~19:13:46, so the filtered/unfiltered
    boundary in the frame list brackets the beads without needing an ephemeris,
    and it survives the frame list being rebuilt at a different cadence. Third
    contact fell in a recording gap, but the rule is written for both edges
    because the last capture before that gap still ends on a bead.
    """
    marked = 0
    for i, f in enumerate(frames):
        if f.get("dense"):
            continue
        near = False
        for j in range(max(0, i - BEAD_FRAMES), min(len(frames), i + BEAD_FRAMES + 1)):
            if frames[j]["state"] != f["state"]:
                near = True
                break
        if near and f["stack"] > 1:
            f["stack"] = 1
            marked += 1
    if marked:
        print("  beads: %d frames within %d of a contact left unstacked"
              % (marked, BEAD_FRAMES))

    # Drop blown frames.
    #
    # Twice during the partials the operator pushed the exposure far up and back
    # again within a capture. Gain normalization scales those frames back down,
    # but clipped highlights do not come back: the disc is saturated solid and
    # renders as a big featureless glow - a "huge blur" a few seconds into the
    # video. A properly exposed filtered frame here saturates 0.000% of its
    # pixels, so anything over a percent is the operator overshooting.
    #
    # Totality is exempt. There the brightest frames are the diamond ring and the
    # exposure ramp after the filter comes off, which are the event itself rather
    # than a mistake.
    # The resolve is exempt: every frame in it clips by construction, and it is
    # the event rather than a mistake. See the note where `resolve` is built.
    blown = [f for f in frames
             if f["satfrac"] > MAX_SATFRAC[f["state"]] and not f.get("resolve")]
    if blown:
        by_file = {}
        for f in blown:
            by_file[f["file"]] = by_file.get(f["file"], 0) + 1
        print("  dropped %d blown filtered frames: %s" % (
            len(blown), ", ".join("%s x%d" % (k, v) for k, v in sorted(by_file.items()))))
        frames = [f for f in frames
                  if f["satfrac"] <= MAX_SATFRAC[f["state"]] or f.get("resolve")]
    for name, sel in (("resolve", lambda f: f.get("resolve")),
                      ("beads", lambda f: f.get("bead")),
                      ("prominence", lambda f: f.get("dense") and not f.get("bead")),
                      ("corona", lambda f: f.get("hold"))):
        g = [f for f in frames if sel(f)]
        if not g:
            continue
        uniq = len(set((f["file"], f["index"]) for f in g))
        print("  %-11s %s f%d-%d -> %d frames (%.1fs) from %d raw (%.1fx)"
              % (name, g[0]["file"], g[0]["index"], g[-1]["index"], len(g),
                 len(g) / args.fps, uniq, len(g) / max(uniq, 1)))
    """
    Flatten each capture before the gains are written out.

    Late on purpose: it needs the final frame list, because it corrects the
    frames that will actually be rendered rather than the segments they came
    from. See flatten_captures.
    """
    flatten_captures(frames)

    for f in frames:
        del f["satfrac"]

    cfg = {
        "frames": frames,
        "outDir": args.frames_dir,
        "width": lc["files"][0]["width"] // 2,   # superpixel demosaic
        "height": lc["files"][0]["height"] // 2,
        # Output is a window cropped around the Sun rather than the full frame:
        # the Sun drifts far enough over 45 minutes that centring it in the full
        # frame would leave a black band down one side.
        "outW": 1280,
        "outH": 720,
        "fps": args.fps,
    }
    cfgdir = os.path.join(args.out, "configs")
    os.makedirs(cfgdir, exist_ok=True)
    dest = os.path.join(cfgdir, "timelapse.json").replace("\\", "/")
    with open(dest, "w") as fh:
        json.dump(cfg, fh)

    n_unf = sum(1 for p in frames if p["state"] == "unfiltered")
    dense = [p for p in frames if p.get("dense")]
    print("%s\n  %d frames (%d unfiltered), %d runs, %dx%d, %.1fs at %dfps" % (
        dest, len(frames), n_unf, len(runs), cfg["width"], cfg["height"],
        len(frames) / args.fps, args.fps))
    if dense:
        uniq = len(set(p["index"] for p in dense))
        print("  prominence level %s f%d-%d resampled to %d frames (%.1fs) from "
              "%d raw, gain %.2f"
              % (dense[0]["file"], dense[0]["index"], dense[-1]["index"],
                 len(dense), len(dense) / args.fps, uniq, dense[0]["gain"]))


# Longest drizzle group, in raw frames. At 23.3 fps this is 0.86 s of effective
# shutter, matching the video's own sampling cadence.
STACK_MAX = 20

# Video frames either side of a filtered/unfiltered boundary that are rendered
# unstacked so the beads survive. At ~0.86 s per frame this is about 9 s each
# way, which comfortably brackets the bead sequence at second contact.
BEAD_FRAMES = 10

# Screen time for the approach half of the second-contact resolve: the plateau
# shrinking, before it breaks into a chain.
RESOLVE_S = 9.0

# Screen time for the bead phase, and the saturation at which the resolve is
# judged to have reached it.
#
# The threshold is MAX_SATFRAC's own unfiltered value, which is the point where a
# frame stops being grossly overexposed and would have passed the ordinary gate -
# so the bead dwell is exactly the part of the resolve that is a normal picture.
#
# It is set wide on purpose. Only about 2.4 s of bead footage exists before the
# operator opened up for the corona, so a ten-second dwell has to repeat frames,
# and how many depends entirely on how much of the tail is claimed: drawn at
# satfrac 0.003 the window is 33 raw frames and every one is held 9.2x, which is
# a slideshow. At 0.02 it is about 70 and the factor drops to ~4. The beads go on
# being visible through the first second of the prominence level after this,
# which is separately held for SLOWMO_LEVEL_S.
# Screen seconds held on the beads.
#
# The window is 67 raw frames, 2.87 s of real time, so this is also a choice of
# how many times each frame repeats: 10 s was a 4.5x hold, about 6.7 fps, which
# reads as hanging on far too long after the beads have gone. 4.5 s is a 2x hold
# and still more than a doubling of real time.
BEADS_S = 4.5

# Raw frames averaged for a bead video frame. See the note in the stacking loop:
# short enough that the beads do not move within it, long enough to take the
# grain off a single short exposure.
BEAD_STACK = 3
BEAD_SATFRAC = MAX_SATFRAC["unfiltered"]

# Screen time for the corona proper - the stable totality levels that are neither
# the resolve nor the prominence exposure.
CORONA_S = 10.0


def stretch_points(seg, fps_out, target_s, lo, hi):
    """Sample raw frames [lo, hi) to fill target_s of screen time.

    Never stops short of the range end, unlike `dense_points` - that bound exists
    to keep a drizzle group from running off the end of a segment, and the two
    callers here are either unstacked (the resolve) or stacked with overlapping
    groups (the corona dwell), so neither needs it.

    Repeats a frame only when the range is shorter than the target asks for. The
    corona has 2,583 raw frames for a ten-second dwell and every video frame is a
    different exposure; the bead phase has about seventy for the same ten seconds,
    so it holds each one for three or four. That is a real limit of the footage -
    two seconds of beads were recorded - and repeating is the honest way to spend
    it, since the alternative is inventing frames that were never exposed.

    The measured fields are INTERPOLATED between the light curve's samples rather
    than taken from the nearest one. The curve is sampled every 20 raw frames, and
    the gain is a function of those numbers, so snapping to the nearest sample
    quantizes the gain into 20-frame plateaus. Everywhere else in the video that
    is invisible - the plateaus are one video frame wide. Here one plateau is two
    thirds of a second of slow motion, and the operator's exposure ramp falls
    inside this segment: the gain climbs 1.85x and then 2.45x between neighbouring
    samples, which without interpolation is two visible steps in brightness.
    """
    src = sorted(seg["points"], key=lambda p: p["i"])
    want = max(2, int(round(target_s * fps_out)))
    num = [k for k, v in src[0].items() if isinstance(v, (int, float))]
    out = []
    for j in range(want):
        i = lo + (j * (hi - lo)) // want
        # Bracketing samples, clamped at both ends of the segment.
        n = next((k for k, p in enumerate(src) if p["i"] > i), len(src))
        b = src[min(n, len(src) - 1)]
        a = src[max(0, n - 1)]
        w = 0.0 if a["i"] == b["i"] else (i - a["i"]) / (b["i"] - a["i"])
        w = min(max(w, 0.0), 1.0)
        p = {k: a[k] + w * (b[k] - a[k]) for k in num}
        p["i"] = i
        p["t"] = a["t"] + (i - a["i"]) / seg["fps"]
        out.append(p)
    return out


def mkframe(seg, point, gain, dense=False, resolve=False, hold=False,
            bead=False):
    f = {
        "src": seg["path"], "file": seg["file"], "index": point["i"],
        "state": seg["state"], "kind": seg["kind"],
        "t_rel": round(point["t"], 3),
        # Segment end, so a drizzle group can be stopped before the exposure changes.
        "seg_end": seg["start"] + seg["count"],
        "gain": round(gain, 8),
        "satfrac": point["satfrac"],
    }
    if dense:
        f["dense"] = True
    if resolve:
        f["resolve"] = True
    if hold:
        f["hold"] = True
    if bead:
        f["bead"] = True
    return f


def dense_points(seg, fps_out, target_s, lo=None, hi=None):
    """One video frame per RAW frame, repeated to fill target_s of screen time.

    Two bounds, and both matter.

    The run stops a full group short of the segment's end. Every frame's drizzle
    group starts at that frame and runs forward, so without this the last twenty
    frames would get progressively shorter groups and the picture would grow
    visibly noisier on its way out - the one place in the video where the noise
    ramps rather than holding steady. Giving up 0.86 s of a scene that is barely
    changing costs nothing by comparison.

    The count is then padded to the target duration. There are 9.9 s of real time
    here against 12 s of screen time, so about seven frames in ten are shown
    twice. That is normally judder, and here it is invisible: the Moon advances
    0.006 px between consecutive raw frames, so adjacent frames differ only by
    seeing and noise, and each is already the mean of twenty. What the viewer sees
    is a very slightly slowed, almost still look at the prominences - which is
    what the footage actually contains.
    """
    src = seg["points"]
    s0, s1 = seg["start"], seg["start"] + seg["count"] - STACK_MAX
    # A caller may ask for part of the segment - the bead window cuts across this
    # one - but never for more of it than the group bound allows.
    lo = s0 if lo is None else max(lo, s0)
    hi = s1 if hi is None else min(hi, s1)
    if hi - lo < 2:
        return [p for p in src if lo <= p["i"] < max(hi, lo + 1)] or src[:1]
    uniq = list(range(lo, hi))
    want = max(len(uniq), int(round(target_s * fps_out)))
    out = []
    for j in range(want):
        i = uniq[min(len(uniq) - 1, int(j * len(uniq) / want))]
        near = min(src, key=lambda p: abs(p["i"] - i))
        p = dict(near)
        p["i"] = i
        p["t"] = near["t"] + (i - near["i"]) / seg["fps"]
        out.append(p)
    return out


if __name__ == "__main__":
    main()
