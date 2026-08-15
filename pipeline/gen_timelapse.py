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


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def level(point):
    """Background level above the black floor, for one sample."""
    return max(point["med"] - point["black"], 1.0)


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

    with open(os.path.join(args.out, "lightcurve.json")) as fh:
        lc = json.load(fh)
    with open(os.path.join(args.out, "segments.json")) as fh:
        man = json.load(fh)

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

        for k, s in enumerate(run):
            gk = gains[k]
            if k == slow:
                for p in dense_points(s, args.fps, SLOWMO_LEVEL_S):
                    frames.append(mkframe(s, p, gk, dense=True))
            elif s["kind"] == "stable":
                # One gain for the whole segment: the camera did not change, so
                # whatever brightness change happens here is the eclipse itself.
                for p in s["points"]:
                    frames.append(mkframe(s, p, gk))
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
                for p in s["points"]:
                    gp = anchor * s["e_in"] / level(p)
                    # A transition frame must never render brighter than the
                    # stable frames it sits between. The exposure proxy moves fast
                    # through a ramp and the estimate overshoots badly at the end
                    # of one - single frames were landing at 3.5 to 4.9 against a
                    # 0.78 target, which is a white flash for one frame. Capping
                    # on the frame's own p99 bounds that directly.
                    cap = TRANSITION_CEILING * TARGET[s["state"]] * FULL_SCALE / max(p["p99"], 1)
                    frames.append(mkframe(s, p, min(gp, cap)))

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
        if f.get("dense"):
            # Dense frames sit one raw frame apart, so the gap-to-the-next rule
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
    blown = [f for f in frames if f["satfrac"] > MAX_SATFRAC[f["state"]]]
    if blown:
        by_file = {}
        for f in blown:
            by_file[f["file"]] = by_file.get(f["file"], 0) + 1
        print("  dropped %d blown filtered frames: %s" % (
            len(blown), ", ".join("%s x%d" % (k, v) for k, v in sorted(by_file.items()))))
        frames = [f for f in frames if f["satfrac"] <= MAX_SATFRAC[f["state"]]]
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


def mkframe(seg, point, gain, dense=False):
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
    return f


def dense_points(seg, fps_out, target_s):
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
    lo, hi = seg["start"], seg["start"] + seg["count"] - STACK_MAX
    if hi - lo < 2:
        return src
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
