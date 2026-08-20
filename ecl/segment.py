"""Stage A — measure the capture and segment it by exposure state.

A single eclipse capture contains several different exposure states: the
white-light filter comes off part way through and the exposure is then ridden
down by hand. Stacking a whole capture would blend all of that together, so this
builds a sampled light curve, finds the state changes, and writes the frame-range
manifest (`segments.json`) that every later pass consumes.

    python -m ecl.segment --data <captures> --out <out>

Nothing here is tuned to a particular file. Change points are found by comparing
each step against the spread of the series' own steps, so a uniform capture
yields exactly one segment and a capture with a filter change yields a boundary
where the brightness actually moves.

WHY THIS IS NOT `scan_ser.py` ANY MORE. It was, and it globbed `*.ser`, opened
each with the SER reader directly, and got its statistics from a method that
only that reader had. That made it the one stage in the pipeline that could not
run on a folder of camera images - and since every later pass reads
`segments.json`, a DSLR user got no further than this. It now goes through
`ecl.source`, like everything else does, so all three accepted layouts work.

Outputs `lightcurve.json` (cached; delete it, or change --step/--rows, to force a
rescan), `segments.json` and a human-readable `segments.txt`.
"""

import argparse
import json
import math
import os
import sys
import time

from . import paths
from .source import discover, open_source

__all__ = ["build_lightcurve", "segment_capture", "classify", "main"]

# Sampling: every Nth frame, and this many evenly spaced rows within each frame.
DEFAULT_STEP = 20
DEFAULT_ROWS = 24

# ...but never so few samples that the change-point test has nothing to work
# with. A SER holds thousands of frames and every 20th is plenty; a camera run
# is often a few hundred files, where every 20th leaves a dozen points and the
# MAD of eleven steps is not a spread. The step is capped per capture so the
# behaviour on a long SER is exactly what it always was.
MIN_SAMPLES = 60

# A segment must last at least this long to count as a stable state; shorter runs
# are the exposure being ridden and get merged into a `transition` segment.
MIN_SEGMENT_S = 2.0

# Change-point sensitivity. A boundary needs a step in log2(median) larger than
# both this floor (a ~27% brightness change) and several times the typical step
# size of the series itself.
CP_FLOOR_LOG2 = 0.35
CP_MAD_MULT = 6.0

# Exposure clustering: stable segments whose log2(median) differ by more than this
# belong to different exposure levels.
LEVEL_GAP_LOG2 = 0.5


# The smallest signal worth taking a logarithm of: one count at 16 bits. Levels
# arrive as fractions of full scale (see `source.frame_stats`), so this is a
# fraction too - it was literally 1, which on normalised float data floors every
# sample in the series to the same value and finds no change points at all.
LEVEL_FLOOR = 1.0 / 65535.0


def log2s(v):
    return math.log(max(v, LEVEL_FLOOR), 2)


def mad(xs):
    """Median absolute deviation — robust spread."""
    if not xs:
        return 0.0
    s = sorted(xs)
    m = s[len(s) // 2]
    d = sorted(abs(x - m) for x in xs)
    return d[len(d) // 2]


def sample_step(frame_count, requested):
    """Sampling stride for one capture: the request, or finer if it must be."""
    return max(1, min(requested, frame_count // MIN_SAMPLES or 1))


# ---------------------------------------------------------------- light curve

def sample_capture(path, step, nrows, log):
    """Sample one capture into a list of per-frame stat dicts."""
    with open_source(path) as src:
        n = src.frame_count
        step = sample_step(n, step)
        ts = src.timestamps()
        t0 = ts[0]
        idx = list(range(0, n, step))
        if idx[-1] != n - 1:
            idx.append(n - 1)

        pts = []
        for i in idx:
            st = src.sample_stats(i, nrows)
            st["i"] = i
            st["t"] = (ts[i] - t0).total_seconds()
            pts.append(st)

        if not getattr(src, "has_real_times", True):
            log("  %s: NO capture timestamps; using file mtimes. Frame order and "
                "pacing are right, absolute UTC is not." % src.name)

        log("  %s: %d frames, %d samples, %.1fs @ %.2ffps" % (
            src.name, n, len(pts), (ts[-1] - t0).total_seconds(), src.fps()))
        return {
            "path": str(path).replace("\\", "/"),
            "name": src.name,
            "width": src.width, "height": src.height, "depth": src.depth,
            "color_id": src.cfa_pattern or 0,
            "frame_count": n,
            "max_value": src.max_value,
            "t0_utc": ts[0].isoformat(), "t1_utc": ts[-1].isoformat(),
            "duration": (ts[-1] - t0).total_seconds(), "fps": src.fps(),
            "step": step, "rows": nrows,
            "points": pts,
        }


def build_lightcurve(data_dir, out_dir, step, nrows, log):
    """The sampled brightness of every capture, cached on disk.

    The cache is keyed on the sampling parameters AND on the list of captures.
    Keying it on the parameters alone means pointing the pipeline at a second
    data set silently reuses the first one's curve, which looks like a working
    run right up to the render.
    """
    caps = discover(data_dir)
    if not caps:
        raise SystemExit(
            "no captures found under %s\n"
            "  expected .ser files, one folder of images per capture, or a "
            "single flat folder of images" % data_dir)
    names = [os.path.basename(str(c).rstrip("/\\")) for c in caps]

    cache = os.path.join(out_dir, "lightcurve.json")
    if os.path.exists(cache):
        try:
            with open(cache, encoding="utf-8") as fh:
                lc = json.load(fh)
        except (ValueError, OSError):
            lc = {}
        if (lc.get("step") == step and lc.get("rows") == nrows
                and [f["name"] for f in lc.get("files", [])] == names):
            log("using cached light curve (%d captures)" % len(lc["files"]))
            return lc
        log("cache does not match this request, rescanning")

    log("scanning %d capture(s), every %dth frame, %d rows/frame"
        % (len(caps), step, nrows))
    started = time.time()
    files = [sample_capture(p, step, nrows, log) for p in caps]
    lc = {"step": step, "rows": nrows,
          "data_dir": str(data_dir).replace("\\", "/"), "files": files}
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump(lc, fh)
    log("light curve built in %.1f min -> %s"
        % ((time.time() - started) / 60, cache))
    return lc


# ------------------------------------------------------------- change points

def find_boundaries(pts):
    """Sample indices where the brightness state changes.

    Returns positions into `pts` at which a new segment begins.
    """
    if len(pts) < 3:
        return []
    series = [log2s(p["med"]) for p in pts]
    steps = [abs(series[i] - series[i - 1]) for i in range(1, len(series))]
    thr = max(CP_FLOOR_LOG2, CP_MAD_MULT * mad(steps))
    return [i for i in range(1, len(series)) if steps[i - 1] > thr]


def refine_boundary(src, nrows, lo, hi, lo_med, hi_med):
    """Binary-search the exact frame where the brightness crosses the midpoint.

    Sampling every 20th frame only locates a transition to within 20 frames; this
    narrows it to the frame, for ~5 extra reads instead of 20.
    """
    target = (log2s(lo_med) + log2s(hi_med)) / 2.0
    rising = hi_med > lo_med
    while hi - lo > 1:
        mid = (lo + hi) // 2
        m = log2s(src.sample_stats(mid, nrows)["med"])
        if (m >= target) == rising:
            hi = mid
        else:
            lo = mid
    return hi


# ------------------------------------------------------------- segmentation

def segment_capture(fdesc, log):
    """Split one capture into frame ranges of constant exposure state."""
    pts = fdesc["points"]
    bounds = find_boundaries(pts)

    # Refine each sampled boundary down to an exact frame index.
    exact = []
    if bounds:
        with open_source(fdesc["path"]) as src:
            for b in bounds:
                lo, hi = pts[b - 1], pts[b]
                exact.append(refine_boundary(src, fdesc["rows"], lo["i"],
                                             hi["i"], lo["med"], hi["med"]))

    edges = [0] + sorted(set(exact)) + [fdesc["frame_count"]]
    fps = fdesc["fps"] or 1.0
    # Half of full scale, as a fraction: the frame is blown when its MEDIAN is
    # up there, which only happens with the filter off and the exposure still
    # set for the photosphere.
    blown_level = 0.5

    segs = []
    for k in range(len(edges) - 1):
        start, end = edges[k], edges[k + 1]
        if end <= start:
            continue
        inside = [p for p in pts if start <= p["i"] < end]
        if not inside:
            inside = [min(pts, key=lambda p: abs(p["i"] - start))]
        med = sorted(p["med"] for p in inside)[len(inside) // 2]
        segs.append({
            "start": start, "count": end - start,
            "seconds": (end - start) / fps,
            "med": med,
            "p99": sorted(p["p99"] for p in inside)[len(inside) // 2],
            "lit": round(sorted(p["lit"] for p in inside)[len(inside) // 2], 5),
            "satfrac": round(sorted(p["satfrac"] for p in inside)[len(inside) // 2], 5),
            "blown": med >= blown_level,
        })

    return merge_short(segs, fps)


def merge_short(segs, fps):
    """Collapse sub-threshold runs into `transition` segments.

    While the operator rides the exposure down after pulling the filter, every
    sample differs from the last. Those frames belong to no stable state, so they
    are grouped rather than emitted as a dozen one-second segments.
    """
    out = []
    for s in segs:
        short = s["seconds"] < MIN_SEGMENT_S and not s["blown"]
        if short and out and out[-1].get("kind") == "transition":
            prev = out[-1]
            prev["count"] = s["start"] + s["count"] - prev["start"]
            prev["seconds"] = prev["count"] / fps
            prev["med"] = max(prev["med"], s["med"])
            continue
        s["kind"] = "transition" if short else ("blown" if s["blown"] else "stable")
        out.append(s)
    return out


def classify(files_segs, log):
    """Label segments filtered/unfiltered and assign exposure levels.

    Totality is bracketed physically rather than by a brightness threshold: pulling
    the filter while the exposure is still set for the photosphere blows the frame
    out, and the sun reappearing at third contact blows it out again. Everything
    between the first and last blowout is unfiltered.
    """
    flat = []
    for f in files_segs:
        for s in f["segments"]:
            flat.append((f, s))

    blown_at = [i for i, (_, s) in enumerate(flat) if s["kind"] == "blown"]
    if blown_at:
        first, last = blown_at[0], blown_at[-1]
        log("totality bracket: %s f%d -> %s f%d" % (
            flat[first][0]["name"], flat[first][1]["start"],
            flat[last][0]["name"], flat[last][1]["start"]))
    else:
        first = last = -1
        log("WARNING: no blown-out transition found; treating all as filtered")

    for i, (_, s) in enumerate(flat):
        s["state"] = "unfiltered" if first <= i <= last else "filtered"

    # Cluster the stable unfiltered segments into exposure levels by background
    # median, splitting wherever a gap in log space exceeds the threshold.
    corona = [s for _, s in flat if s["state"] == "unfiltered" and s["kind"] == "stable"]
    ordered = sorted(corona, key=lambda s: s["med"])
    level = 0
    for i, s in enumerate(ordered):
        if i and log2s(s["med"]) - log2s(ordered[i - 1]["med"]) > LEVEL_GAP_LOG2:
            level += 1
        s["level"] = level
    names = {0: "short", 1: "mid", 2: "long"} if level == 2 else {}
    for s in ordered:
        s["level_name"] = names.get(s["level"], "L%d" % s["level"])
    log("exposure levels among corona segments: %d" % (level + 1))
    return flat


# --------------------------------------------------------------------- report

def write_report(files_segs, out_dir, log):
    lines = ["%-14s %-6s %-7s %-6s %-11s %-6s %6s %9s %9s" % (
        "file", "start", "count", "sec", "state", "kind", "level", "med", "p99"),
        "-" * 90]
    for f in files_segs:
        for s in f["segments"]:
            # med and p99 are fractions of full scale, so the table reads the
            # same whether the camera was 8-bit, 16-bit or float.
            lines.append("%-14s %-6d %-7d %-6.1f %-11s %-6s %6s %9.5f %9.5f" % (
                f["name"], s["start"], s["count"], s["seconds"], s["state"],
                s["kind"], s.get("level_name", "-"), s["med"], s["p99"]))
    txt = os.path.join(out_dir, "segments.txt")
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log("wrote %s (%d segments)"
        % (txt, sum(len(f["segments"]) for f in files_segs)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=paths.data_dir())
    ap.add_argument("--out", default=paths.out_dir())
    ap.add_argument("--step", type=int, default=DEFAULT_STEP,
                    help="sample every Nth frame (finer on short captures)")
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS,
                    help="rows sampled per frame")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "scan.log"), "w", encoding="utf-8") as logfh:

        def log(msg):
            logfh.write(msg + "\n")
            logfh.flush()
            print(" ", msg)

        lc = build_lightcurve(args.data, args.out, args.step, args.rows, log)

        files_segs = []
        for f in lc["files"]:
            files_segs.append({
                "name": f["name"], "path": f["path"],
                "frame_count": f["frame_count"], "fps": f["fps"],
                "t0_utc": f["t0_utc"], "t1_utc": f["t1_utc"],
                "color_id": f["color_id"], "width": f["width"],
                "height": f["height"], "depth": f["depth"],
                "segments": segment_capture(f, log),
            })

        classify(files_segs, log)

        manifest = os.path.join(args.out, "segments.json")
        with open(manifest, "w", encoding="utf-8") as fh:
            json.dump({"data_dir": lc["data_dir"], "files": files_segs}, fh,
                      indent=1)
        write_report(files_segs, args.out, log)

    print("segments -> %s" % manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
